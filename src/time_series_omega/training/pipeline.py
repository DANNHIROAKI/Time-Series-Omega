"""Training utilities for SFF-Ω."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.datasets import SequenceBatch
from ..models.sff_omega import SFFOmega
from ..segmentation import mdl_penalty


@dataclass
class AdversarialConfig:
    enabled: bool = False
    epsilon: float = 0.05
    epsilon_log: float = 0.05
    steps: int = 3
    step_size: float = 0.02
    logsumexp_temperature: float = 0.01
    penalty_weight: float = 0.0
    include_clean: bool = True


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    smoothness_weight: float = 1.0
    moments_weight: float = 1.0
    soft_anchor_weight: float = 1.0
    consensus_weight: float = 0.0
    calendar_weight: float = 0.1
    h_disc_weight: float = 0.1
    mdl_structure_weight: float = 0.0
    mdl_curvature_weight: float = 0.0
    mdl_log_derivative_weight: float = 0.0
    grad_clip: Optional[float] = 1.0
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)


class Trainer:
    def __init__(self, model: SFFOmega, config: TrainConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.loss_fn = nn.MSELoss()

    def _adversarial_perturb(
        self,
        series: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        if not self.config.adversarial.enabled:
            return series, []
        adv = series.clone().detach()
        epsilon = self.config.adversarial.epsilon
        epsilon_log = self.config.adversarial.epsilon_log
        step_size = self.config.adversarial.step_size
        lower = series.amin()
        upper = series.amax()
        perturbations: List[torch.Tensor] = []
        for _ in range(self.config.adversarial.steps):
            adv = adv.detach().requires_grad_(True)
            pred = self.model(adv, mask=mask)
            loss = self.loss_fn(pred, target)
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
            adv = adv + step_size * torch.sign(grad)
            delta = adv - series
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv = torch.clamp(series + delta, lower - epsilon_log, upper + epsilon_log)
            if mask is not None:
                adv = adv * mask
            perturbations.append(adv.detach())
        if not perturbations:
            return series, []
        return perturbations[-1], perturbations

    def step(self, batch: SequenceBatch, template: Optional[torch.Tensor] = None) -> Dict[str, float]:
        self.model.train()
        series = batch.values.to(self.config.device)
        covariates = None if batch.covariates is None else batch.covariates.to(self.config.device)
        mask = None if batch.mask is None else batch.mask.to(self.config.device)
        target = series[:, -self.model.horizon :]
        adv_series, perturbations = self._adversarial_perturb(series, target, mask)
        pred, gauge = self.model(adv_series, covariates=covariates, mask=mask, return_gauge=True)
        loss = self.loss_fn(pred, target)
        regs = self.model.regularisation(adv_series, covariates=covariates, mask=mask, template=template, cohort=gauge.canonical)
        loss = loss + self.config.smoothness_weight * regs["smoothness"]
        loss = loss + self.config.moments_weight * regs["moments"]
        if "soft_anchor" in regs:
            loss = loss + self.config.soft_anchor_weight * regs["soft_anchor"]
        if "consensus" in regs and self.config.consensus_weight > 0:
            loss = loss + self.config.consensus_weight * regs["consensus"]
        if "calendar" in regs:
            loss = loss + self.config.calendar_weight * regs["calendar"]
        if "h_disc" in regs:
            loss = loss + self.config.h_disc_weight * regs["h_disc"]
        mdl_terms = mdl_penalty(self.model.time_warp, self.model.value_transform, series.shape[1])
        loss = loss + self.config.mdl_structure_weight * mdl_terms["mdl_structure"]
        loss = loss + self.config.mdl_curvature_weight * mdl_terms["mdl_curvature"]
        loss = loss + self.config.mdl_log_derivative_weight * mdl_terms["mdl_log_derivative"]
        adv_penalty_value: Optional[torch.Tensor] = None
        adv_cfg = self.config.adversarial
        if adv_cfg.enabled and adv_cfg.penalty_weight > 0.0:
            if adv_cfg.include_clean or perturbations:
                candidate_losses = [loss]
                if adv_cfg.include_clean:
                    clean_pred = self.model(series, covariates=covariates, mask=mask)
                    candidate_losses.append(self.loss_fn(clean_pred, target))
                for idx, pert in enumerate(perturbations):
                    if idx == len(perturbations) - 1:
                        continue
                    pert_pred = self.model(pert, covariates=covariates, mask=mask)
                    candidate_losses.append(self.loss_fn(pert_pred, target))
                losses_tensor = torch.stack(candidate_losses)
                temperature = max(adv_cfg.logsumexp_temperature, 1e-6)
                adv_penalty_value = temperature * torch.logsumexp(losses_tensor / temperature, dim=0)
                loss = loss + adv_cfg.penalty_weight * adv_penalty_value
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        metrics = {"loss": float(loss.item())}
        metrics.update({k: float(v.detach().item()) for k, v in regs.items()})
        metrics.update({k: float(v.detach().item()) for k, v in mdl_terms.items()})
        if adv_penalty_value is not None:
            metrics["adv_penalty"] = float(adv_penalty_value.detach().item())
        return metrics

    def fit(self, loader: DataLoader[SequenceBatch], template: Optional[torch.Tensor] = None) -> None:
        for _ in range(self.config.epochs):
            for batch in loader:
                self.step(batch, template)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[SequenceBatch]) -> Dict[str, float]:
        self.model.eval()
        mae_total = 0.0
        mse_total = 0.0
        denom = 0
        for batch in loader:
            series = batch.values.to(self.config.device)
            covariates = None if batch.covariates is None else batch.covariates.to(self.config.device)
            mask = None if batch.mask is None else batch.mask.to(self.config.device)
            target = series[:, -self.model.horizon :]
            pred = self.model(series, covariates=covariates, mask=mask)
            diff = pred - target
            mae_total += diff.abs().sum().item()
            mse_total += diff.pow(2).sum().item()
            denom += diff.numel()
        mae = mae_total / denom if denom else 0.0
        rmse = (mse_total / denom) ** 0.5 if denom else 0.0
        return {"mae": mae, "rmse": rmse}

    @torch.no_grad()
    def canonical_residuals(
        self, loader: DataLoader[SequenceBatch]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Collect canonical-domain residuals for conformal calibration."""

        self.model.eval()
        residuals: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        has_mask = False
        target_device = torch.device(self.config.device)
        for batch in loader:
            series = batch.values.to(self.config.device)
            covariates = None if batch.covariates is None else batch.covariates.to(self.config.device)
            mask = None if batch.mask is None else batch.mask.to(self.config.device)
            _, gauge = self.model(series, covariates=covariates, mask=mask, return_gauge=True)
            if gauge.canonical_prediction is None:
                raise RuntimeError("model must be run with return_gauge=True to collect residuals")
            residuals.append(gauge.canonical - gauge.canonical_prediction)
            if mask is not None:
                masks.append(mask)
                has_mask = True
        if residuals:
            residual_tensor = torch.cat(residuals, dim=0)
        else:
            residual_tensor = torch.empty(0, device=target_device)
        mask_tensor = torch.cat(masks, dim=0) if has_mask and masks else None
        return residual_tensor, mask_tensor


@dataclass
class GateDecision:
    enabled: bool
    risk_delta: float
    threshold: float


class DeploymentGate:
    """Risk-difference gate based on a validation set."""

    def __init__(self, delta: float = 0.05) -> None:
        self.delta = delta

    def decide(
        self,
        candidate_losses: torch.Tensor,
        baseline_losses: torch.Tensor,
    ) -> GateDecision:
        if candidate_losses.shape != baseline_losses.shape:
            raise ValueError("loss tensors must share the same shape")
        diff = candidate_losses - baseline_losses
        mean = diff.mean().item()
        variance = diff.var(unbiased=False).item()
        neff = max(diff.numel(), 1)
        spi = math.sqrt(2.0 * variance * math.log(2.0 / max(self.delta, 1e-6)) / neff)
        threshold = mean + spi
        enabled = threshold <= 0.0
        return GateDecision(enabled=enabled, risk_delta=mean, threshold=threshold)


__all__ = ["Trainer", "TrainConfig", "DeploymentGate", "GateDecision"]
