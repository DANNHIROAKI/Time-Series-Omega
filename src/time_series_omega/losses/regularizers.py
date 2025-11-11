"""Regularisation terms used in the SFF-Ω pipeline."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


class SlidingWindowMoments(nn.Module):
    """Compute mean/variance deviations across sliding windows."""

    def __init__(self, window_sizes: Iterable[int], reduction: str = "mean") -> None:
        super().__init__()
        self.window_sizes = list(window_sizes)
        self.reduction = reduction

    def forward(
        self,
        series: torch.Tensor,
        target_mean: float = 0.0,
        target_var: float = 1.0,
    ) -> torch.Tensor:
        total = []
        for window in self.window_sizes:
            if window <= 1:
                continue
            kernel = torch.ones(1, 1, window, device=series.device) / window
            mean = torch.nn.functional.conv1d(
                series.transpose(1, 2), kernel, padding=window // 2
            ).transpose(1, 2)
            var = torch.nn.functional.conv1d(
                (series - mean).transpose(1, 2) ** 2,
                kernel,
                padding=window // 2,
            ).transpose(1, 2)
            total.append((mean - target_mean) ** 2 + (var - target_var) ** 2)
        if not total:
            return torch.tensor(0.0, device=series.device)
        stacked = torch.stack(total, dim=0)
        if self.reduction == "mean":
            return stacked.mean()
        return stacked.sum()


class STFTSoftAnchor(nn.Module):
    """Soft anchor using a distance in the time–frequency plane."""

    def __init__(self, n_fft: int = 64, hop_length: Optional[int] = None, power: float = 2.0) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.power = power

    def forward(
        self,
        series: torch.Tensor,
        template: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        stft = torch.stft(
            series.squeeze(-1), self.n_fft, hop_length=self.hop_length, return_complex=True
        )
        energy = stft.abs() ** self.power
        diff = energy - template
        if weight is not None:
            diff = diff * weight
        return diff.pow(2).mean()


def time_warp_penalty(
    warp_module: nn.Module,
    curvature_weight: float = 1.0,
    log_derivative_weight: float = 1.0,
) -> torch.Tensor:
    curvature = warp_module.curvature_penalty()
    log_derivative = warp_module.log_derivative_penalty()
    return curvature_weight * curvature + log_derivative_weight * log_derivative


def consensus_energy(series: torch.Tensor) -> torch.Tensor:
    """Pairwise energy encouraging aligned canonical trajectories."""

    if series.dim() != 3:
        raise ValueError("series must have shape (batch, length, channels)")
    diffs = series[:, None, :, :] - series[None, :, :, :]
    return (diffs.pow(2).mean(dim=(-2, -1))).mean()


__all__ = [
    "SlidingWindowMoments",
    "STFTSoftAnchor",
    "time_warp_penalty",
    "consensus_energy",
]
