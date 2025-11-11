"""Training utilities for SFF-Ω."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..data.datasets import SequenceBatch
from ..models.sff_omega import SFFOmega


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
    grad_clip: Optional[float] = 1.0


class Trainer:
    def __init__(self, model: SFFOmega, config: TrainConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.loss_fn = nn.MSELoss()

    def step(self, batch: SequenceBatch, template: Optional[torch.Tensor] = None) -> Dict[str, float]:
        self.model.train()
        series = batch.values.to(self.config.device)
        target = series[:, -self.model.horizon :]
        pred, gauge = self.model(series, return_gauge=True)
        loss = self.loss_fn(pred, target)
        regs = self.model.regularisation(series, template, cohort=gauge.canonical)
        loss = loss + self.config.smoothness_weight * regs["smoothness"]
        loss = loss + self.config.moments_weight * regs["moments"]
        if "soft_anchor" in regs:
            loss = loss + self.config.soft_anchor_weight * regs["soft_anchor"]
        if "consensus" in regs and self.config.consensus_weight > 0:
            loss = loss + self.config.consensus_weight * regs["consensus"]
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        metrics = {"loss": float(loss.item())}
        metrics.update({k: float(v.detach().item()) for k, v in regs.items()})
        return metrics

    def fit(self, loader: DataLoader[SequenceBatch], template: Optional[torch.Tensor] = None) -> None:
        for epoch in range(self.config.epochs):
            for batch in loader:
                metrics = self.step(batch, template)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[SequenceBatch]) -> Dict[str, float]:
        self.model.eval()
        mae_total = 0.0
        mse_total = 0.0
        denom = 0
        for batch in loader:
            series = batch.values.to(self.config.device)
            target = series[:, -self.model.horizon :]
            pred = self.model(series)
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
        std = diff.std(unbiased=False).item()
        scale = math.sqrt(2.0 / max(diff.numel(), 1))
        threshold = mean + std * scale
        enabled = threshold <= 0.0 or mean <= -self.delta * abs(baseline_losses.mean().item())
        return GateDecision(enabled=enabled, risk_delta=mean, threshold=threshold)


__all__ = ["Trainer", "TrainConfig", "DeploymentGate", "GateDecision"]
