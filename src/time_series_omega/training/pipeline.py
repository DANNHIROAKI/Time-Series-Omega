"""Training utilities for SFF-Ω."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.datasets import SequenceBatch
from ..models.sff_omega import SFFOmega
from ..robustness.diffeo import (
    DiffeomorphicAdversary,
    DiffeomorphismConstraints,
)


@dataclass
class AdversarialConfig:
    enabled: bool = False
    epsilon: float = 0.05
    epsilon_log: float = 0.05
    steps: int = 3
    step_size: float = 0.02
    cutoff: float = 0.45


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
    mdl_structure_weight: float = 0.1
    mdl_curvature_weight: float = 0.1
    mdl_log_derivative_weight: float = 0.1
    grad_clip: Optional[float] = 1.0
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)


class Trainer:
    def __init__(self, model: SFFOmega, config: TrainConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.loss_fn = nn.MSELoss()
        self._adversary = None
        if config.adversarial.enabled:
            constraints = DiffeomorphismConstraints(
                epsilon_inf=config.adversarial.epsilon,
                epsilon_log=config.adversarial.epsilon_log,
            )
            self._adversary = DiffeomorphicAdversary(
                steps=config.adversarial.steps,
                step_size=config.adversarial.step_size,
                cutoff=config.adversarial.cutoff,
                constraints=constraints,
            )

    def _adversarial_perturb(
        self,
        series: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.config.adversarial.enabled:
            return series
        if self._adversary is None:
            return series
        target = series[:, -self.model.horizon :]
        return self._adversary.perturb(self.model, series, mask=mask, target=target)

    def step(self, batch: SequenceBatch, template: Optional[torch.Tensor] = None) -> Dict[str, float]:
        self.model.train()
        series = batch.values.to(self.config.device)
        covariates = None if batch.covariates is None else batch.covariates.to(self.config.device)
        mask = None if batch.mask is None else batch.mask.to(self.config.device)
        target = series[:, -self.model.horizon :]
        adv_series = self._adversarial_perturb(series, mask)
        pred, gauge = self.model(adv_series, covariates=covariates, mask=mask, return_gauge=True)
        loss = self.loss_fn(pred, target)
        regs = self.model.regularisation(adv_series, covariates=covariates, mask=mask, template=template, cohort=gauge.canonical)
        loss = loss + self.config.smoothness_weight * regs["smoothness"]
        loss = loss + self.config.moments_weight * regs["moments"]
        loss = loss + self.config.mdl_structure_weight * regs["mdl_structure"]
        loss = loss + self.config.mdl_curvature_weight * regs["mdl_curvature"]
        loss = loss + self.config.mdl_log_derivative_weight * regs["mdl_log_derivative"]
        if "soft_anchor" in regs:
            loss = loss + self.config.soft_anchor_weight * regs["soft_anchor"]
        if "consensus" in regs and self.config.consensus_weight > 0:
            loss = loss + self.config.consensus_weight * regs["consensus"]
        if "calendar" in regs:
            loss = loss + self.config.calendar_weight * regs["calendar"]
        if "h_disc" in regs:
            loss = loss + self.config.h_disc_weight * regs["h_disc"]
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        metrics = {"loss": float(loss.item())}
        metrics.update({k: float(v.detach().item()) for k, v in regs.items()})
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
