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


@dataclass
class AdversarialConfig:
    enabled: bool = False
    epsilon: float = 0.05
    epsilon_log: float = 0.05
    steps: int = 3
    step_size: float = 0.02


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
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.config.adversarial.enabled:
            return series
        adv = series.clone().detach().requires_grad_(True)
        epsilon = self.config.adversarial.epsilon
        epsilon_log = self.config.adversarial.epsilon_log
        step_size = self.config.adversarial.step_size
        for _ in range(self.config.adversarial.steps):
            pred = self.model(adv, mask=mask)
            target = series[:, -self.model.horizon :]
            loss = self.loss_fn(pred, target)
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
            adv = adv + step_size * torch.sign(grad)
            delta = adv - series
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv = torch.clamp(series + delta, series.min() - epsilon_log, series.max() + epsilon_log)
            if mask is not None:
                adv = adv * mask
            adv = adv.detach().requires_grad_(True)
        return adv.detach()

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
