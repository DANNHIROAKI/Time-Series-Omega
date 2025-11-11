"""Implementation of the SFF-Ω pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import nn

from ..losses.regularizers import (
    STFTSoftAnchor,
    SlidingWindowMoments,
    consensus_energy,
    low_pass_filter,
    time_warp_penalty,
)
from ..segmentation.mdl import mdl_penalty
from ..transforms.calendar import CalendarAlignment
from ..transforms.time_warp import MonotoneTimeWarp, WarpDiagnostics
from ..transforms.value_transform import MonotoneRQTransform
from .residual import LipschitzMLP
from .ssm import StableSSM


@dataclass
class GaugeOutputs:
    warped: torch.Tensor
    canonical: torch.Tensor
    covariates: Optional[torch.Tensor] = None
    canonical_prediction: Optional[torch.Tensor] = None
    stft_template: Optional[torch.Tensor] = None
    warp_diagnostics: Optional[WarpDiagnostics] = None
    grid: Optional[torch.Tensor] = None
    filtered: Optional[torch.Tensor] = None


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
        calendar_features: Optional[Sequence[str]] = None,
        calendar_period: Optional[int] = None,
        h_disc_cutoff: float = 0.45,
    ) -> None:
        super().__init__()
        self.time_warp = MonotoneTimeWarp(length)
        self.value_transform = MonotoneRQTransform(channels)
        self.sliding_moments = SlidingWindowMoments(window_sizes=[16, 32, 64])
        self.soft_anchor = STFTSoftAnchor(n_fft=n_fft)
        self.ssm = StableSSM(channels, hidden_dim, channels, memory=memory)
        res_hidden = [] if residual_hidden is None else [residual_hidden, residual_hidden]
        self.residual = LipschitzMLP(channels * memory, res_hidden or [channels], channels)
        self.calendar = (
            CalendarAlignment(calendar_features, period=calendar_period)
            if calendar_features is not None
            else None
        )
        self.horizon = horizon
        self.memory = memory
        self.h_disc_cutoff = float(h_disc_cutoff)

    def canonicalise(
        self,
        series: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> GaugeOutputs:
        filtered = low_pass_filter(series, cutoff=self.h_disc_cutoff)
        warped_series = self.time_warp.apply(filtered, mask=mask)
        canonical = self.value_transform(warped_series)
        aligned_covariates = None
        if covariates is not None and self.calendar is not None:
            aligned_covariates = self.calendar(covariates)
        grid, diagnostics = self.time_warp.forward(return_diagnostics=True)  # type: ignore[misc]
        return GaugeOutputs(
            warped=warped_series,
            canonical=canonical,
            covariates=aligned_covariates,
            warp_diagnostics=diagnostics,
            grid=grid,
            filtered=filtered,
        )

    def forward(
        self,
        series: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_gauge: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, GaugeOutputs]:  # type: ignore[override]
        gauge = self.canonicalise(series, covariates=covariates, mask=mask)
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
        raw_reconstruction = self.value_transform.inverse(canonical_pred)
        raw_reconstruction = self.time_warp.inverse_apply(raw_reconstruction, mask=mask)
        forecast = raw_reconstruction[:, -self.horizon :]
        if return_gauge:
            gauge.canonical_prediction = canonical_pred
            return forecast, gauge
        return forecast

    def regularisation(
        self,
        series: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        template: Optional[torch.Tensor] = None,
        cohort: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gauge = self.canonicalise(series, covariates=covariates, mask=mask)
        reg = {
            "smoothness": time_warp_penalty(self.time_warp),
            "moments": self.sliding_moments(gauge.canonical.unsqueeze(-1)),
        }
        mdl_terms = mdl_penalty(
            self.time_warp,
            self.value_transform,
            length=gauge.canonical.shape[1],
        )
        reg.update(mdl_terms)
        if template is not None:
            reg["soft_anchor"] = self.soft_anchor(gauge.canonical.unsqueeze(-1), template)
        if cohort is not None:
            reg["consensus"] = consensus_energy(cohort)
        if gauge.covariates is not None and self.calendar is not None:
            reg["calendar"] = self.calendar.regulariser()
        if gauge.filtered is not None:
            reg["h_disc"] = torch.mean((series - gauge.filtered) ** 2)
        else:
            reg["h_disc"] = torch.tensor(0.0, device=series.device)
        return reg

    def map_canonical_to_raw(
        self,
        canonical_series: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map a canonical-domain series back to the raw observation space."""

        raw = self.value_transform.inverse(canonical_series)
        return self.time_warp.inverse_apply(raw, mask=mask)


__all__ = ["SFFOmega", "GaugeOutputs"]
