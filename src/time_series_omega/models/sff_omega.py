"""Implementation of the SFF-Ω pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from ..losses.regularizers import (
    STFTSoftAnchor,
    SlidingWindowMoments,
    consensus_energy,
    time_warp_penalty,
)
from ..transforms.time_warp import MonotoneTimeWarp
from ..transforms.value_transform import MonotoneRQTransform
from .residual import LipschitzMLP
from .ssm import StableSSM


@dataclass
class GaugeOutputs:
    warped: torch.Tensor
    canonical: torch.Tensor
    canonical_prediction: Optional[torch.Tensor] = None
    stft_template: Optional[torch.Tensor] = None


class SFFOmega(nn.Module):
    """End-to-end SFF-Ω model."""

    def __init__(
        self,
        length: int,
        channels: int,
        horizon: int,
        hidden_dim: int = 32,
        memory: int = 4,
        residual_hidden: Optional[int] = 64,
        n_fft: int = 64,
    ) -> None:
        super().__init__()
        self.time_warp = MonotoneTimeWarp(length)
        self.value_transform = MonotoneRQTransform(channels)
        self.sliding_moments = SlidingWindowMoments(window_sizes=[16, 32, 64])
        self.soft_anchor = STFTSoftAnchor(n_fft=n_fft)
        self.ssm = StableSSM(channels, hidden_dim, channels, memory=memory)
        res_hidden = [] if residual_hidden is None else [residual_hidden, residual_hidden]
        self.residual = LipschitzMLP(channels * memory, res_hidden or [channels], channels)
        self.horizon = horizon
        self.memory = memory

    def canonicalise(self, series: torch.Tensor) -> GaugeOutputs:
        warped_series = self.time_warp.apply(series)
        canonical = self.value_transform(warped_series)
        return GaugeOutputs(warped_series, canonical)

    def forward(
        self,
        series: torch.Tensor,
        return_gauge: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, GaugeOutputs]:  # type: ignore[override]
        gauge = self.canonicalise(series)
        ssm_out = self.ssm(gauge.canonical)
        stacked = []
        padded = torch.nn.functional.pad(gauge.canonical, (0, 0, self.memory - 1, 0))
        for t in range(ssm_out.shape[1]):
            window = []
            for lag in range(self.memory):
                window.append(padded[:, t + self.memory - 1 - lag])
            stacked.append(torch.cat(window, dim=-1))
        stacked_inputs = torch.stack(stacked, dim=1)
        residual = self.residual(stacked_inputs)
        canonical_pred = ssm_out + residual
        pred_tail = canonical_pred[:, -self.horizon :]
        forecast = self.value_transform.inverse(pred_tail)
        if return_gauge:
            return forecast, GaugeOutputs(gauge.warped, gauge.canonical, canonical_pred)
        return forecast

    def regularisation(
        self,
        series: torch.Tensor,
        template: Optional[torch.Tensor] = None,
        cohort: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gauge = self.canonicalise(series)
        reg = {
            "smoothness": time_warp_penalty(self.time_warp),
            "moments": self.sliding_moments(gauge.canonical.unsqueeze(-1)),
        }
        if template is not None:
            reg["soft_anchor"] = self.soft_anchor(gauge.canonical.unsqueeze(-1), template)
        if cohort is not None:
            reg["consensus"] = consensus_energy(cohort)
        return reg


__all__ = ["SFFOmega", "GaugeOutputs"]
