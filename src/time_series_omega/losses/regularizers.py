"""Regularisation terms used in the SFF-Ω pipeline."""
from __future__ import annotations

import math
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
            transposed = series.transpose(1, 2)
            mean_conv = torch.nn.functional.conv1d(transposed, kernel, padding=window // 2)
            mean_conv = mean_conv[..., : series.shape[1]]
            mean = mean_conv.transpose(1, 2)
            centred = series - mean
            var_conv = torch.nn.functional.conv1d(
                centred.transpose(1, 2) ** 2,
                kernel,
                padding=window // 2,
            )
            var_conv = var_conv[..., : series.shape[1]]
            var = var_conv.transpose(1, 2)
            total.append((mean - target_mean) ** 2 + (var - target_var) ** 2)
        if not total:
            return torch.tensor(0.0, device=series.device)
        stacked = torch.stack(total, dim=0)
        if self.reduction == "mean":
            return stacked.mean()
        return stacked.sum()


class STFTSoftAnchor(nn.Module):
    """Soft anchor using a distance in the time–frequency plane."""

    def __init__(
        self,
        n_fft: int = 64,
        hop_length: Optional[int] = None,
        power: float = 2.0,
        log_amplitude: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.power = power
        self.log_amplitude = log_amplitude

    def forward(
        self,
        series: torch.Tensor,
        template: Optional[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if series.dim() == 1:
            series = series.unsqueeze(0)
        if series.dim() == 2:
            flat_series = series
            batch_channels = series.shape[0]
        elif series.dim() == 3:
            batch, length, channels = series.shape
            flat_series = series.permute(0, 2, 1).reshape(-1, length)
            batch_channels = batch * channels
        else:
            raise ValueError("series must have shape (batch, length[, channels])")

        stft = torch.stft(
            flat_series,
            self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )
        energy = stft.abs() ** self.power
        if self.log_amplitude:
            energy = torch.log1p(energy)
            if template is not None:
                template = torch.log1p(template)

        if template is None:
            template = torch.zeros_like(energy)
        else:
            if template.dim() == 3:
                template = template.reshape(batch_channels, *template.shape[-2:])
            if template.shape != energy.shape:
                raise ValueError("template shape must match STFT energy shape")

        diff = energy - template
        if weight is not None:
            if weight.shape != diff.shape:
                raise ValueError("weight must match diff shape")
            diff = diff * weight
        return diff.pow(2).mean()


def time_warp_penalty(
    warp_module: nn.Module,
    curvature_weight: float = 1.0,
    log_derivative_weight: float = 1.0,
    component_weight: float = 1e-2,
) -> torch.Tensor:
    curvature = warp_module.curvature_penalty()
    log_derivative = warp_module.log_derivative_penalty()
    total = curvature_weight * curvature + log_derivative_weight * log_derivative
    if hasattr(warp_module, "complexity"):
        complexity = warp_module.complexity()
        for name, value in complexity.items():
            if name in {"curvature", "log_derivative"}:
                continue
            total = total + component_weight * value
    return total


def consensus_energy(series: torch.Tensor) -> torch.Tensor:
    """Pairwise energy encouraging aligned canonical trajectories."""

    if series.dim() != 3:
        raise ValueError("series must have shape (batch, length, channels)")
    diffs = series[:, None, :, :] - series[None, :, :, :]
    return (diffs.pow(2).mean(dim=(-2, -1))).mean()


def low_pass_filter(series: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Apply an ideal low-pass filter used by the H-disc constraint."""

    if not 0 < cutoff <= 0.5:
        raise ValueError("cutoff must be in (0, 0.5]")
    freq = torch.fft.rfft(series, dim=-2)
    length = series.shape[-2]
    max_bin = int(math.floor(cutoff * length))
    mask = torch.zeros_like(freq, dtype=torch.bool)
    mask[..., : max_bin + 1, :] = True
    filtered = torch.fft.irfft(freq * mask, n=length, dim=-2)
    return filtered


__all__ = [
    "SlidingWindowMoments",
    "STFTSoftAnchor",
    "time_warp_penalty",
    "consensus_energy",
    "low_pass_filter",
]
