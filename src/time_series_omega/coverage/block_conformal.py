"""Block conformal calibration utilities for SFF-Ω."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from ..models.sff_omega import GaugeOutputs, SFFOmega


@dataclass
class ConformalInterval:
    """Interval produced by the block conformal calibrator."""

    lower: torch.Tensor
    upper: torch.Tensor
    width: torch.Tensor


class BlockConformalCalibrator:
    """Canonical-domain block conformal calibrator.

    The calibrator operates on residuals expressed in the canonical domain and
    follows the Ω-2′/Ω-3′ guidelines: residuals are aggregated in non-overlapping
    blocks, the empirical (1-α)-quantile is extracted, and a β-mixing correction
    term is added to guarantee conservative coverage for finite samples.
    """

    def __init__(
        self,
        model: SFFOmega,
        *,
        block_length: int = 32,
        alpha: float = 0.1,
        beta_mixing: Optional[Sequence[float]] = None,
    ) -> None:
        if block_length <= 0:
            raise ValueError("block_length must be positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.model = model
        self.block_length = int(block_length)
        self.alpha = float(alpha)
        self.beta_mixing = tuple(beta_mixing) if beta_mixing is not None else None
        self._quantile: Optional[torch.Tensor] = None
        self._mixing_correction: float = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        residuals: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Estimate the calibration quantile from canonical residuals."""

        if residuals.dim() != 3:
            raise ValueError("residuals must have shape (n, length, channels)")
        abs_residuals = residuals.abs()
        device = residuals.device
        if mask is not None:
            broadcast_mask = self._broadcast_mask(mask, residuals.shape, device)
            abs_residuals = abs_residuals * broadcast_mask
        length = abs_residuals.shape[1]
        if length < self.block_length:
            raise ValueError("sequence length must be >= block_length")
        n_blocks = length // self.block_length
        trimmed = abs_residuals[:, : n_blocks * self.block_length]
        block_scores = trimmed.view(
            trimmed.shape[0],
            n_blocks,
            self.block_length,
            trimmed.shape[-1],
        )
        if mask is not None:
            broadcast_mask = self._broadcast_mask(mask, residuals.shape, device)
            trimmed_mask = broadcast_mask[:, : n_blocks * self.block_length]
            mask_blocks = trimmed_mask.view(
                trimmed_mask.shape[0],
                n_blocks,
                self.block_length,
                trimmed_mask.shape[-1],
            )
            counts = mask_blocks.sum(dim=2).clamp_min(1.0)
            block_scores = block_scores.sum(dim=2) / counts
        else:
            block_scores = block_scores.mean(dim=2)
        flattened = block_scores.reshape(-1, block_scores.shape[-1])
        quantile = torch.quantile(flattened, 1.0 - self.alpha, dim=0)
        quantile = torch.nan_to_num(quantile, nan=0.0, posinf=0.0, neginf=0.0)
        self._quantile = quantile
        self._mixing_correction = self._compute_mixing_correction()

    def _broadcast_mask(
        self, mask: torch.Tensor, target_shape: Tuple[int, int, int], device: torch.device
    ) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        if mask.shape != target_shape:
            raise ValueError("mask must broadcast to residual shape")
        return mask.to(dtype=torch.float32, device=device)

    def _compute_mixing_correction(self) -> float:
        if self.beta_mixing is None:
            return 0.0
        tail_start = min(len(self.beta_mixing), self.block_length)
        tail = self.beta_mixing[tail_start:]
        return float(sum(tail))

    # ------------------------------------------------------------------
    # Interval construction
    # ------------------------------------------------------------------
    def calibrate(
        self,
        gauge: GaugeOutputs,
        *,
        mask: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
    ) -> ConformalInterval:
        """Return calibrated prediction intervals in the raw domain."""

        if self._quantile is None:
            raise RuntimeError("calibrator has not been fitted yet")
        if gauge.canonical_prediction is None:
            raise ValueError("gauge output is missing canonical predictions")
        device = gauge.canonical_prediction.device
        horizon = horizon or self.model.horizon
        inflation = self._quantile.to(device) + self._mixing_correction
        inflation = inflation.view(1, 1, -1)
        canonical_lower = gauge.canonical_prediction - inflation
        canonical_upper = gauge.canonical_prediction + inflation
        mask_tensor = None if mask is None else mask.to(device)
        raw_lower_full = self.model.map_canonical_to_raw(canonical_lower, mask=mask_tensor)
        raw_upper_full = self.model.map_canonical_to_raw(canonical_upper, mask=mask_tensor)
        lower = raw_lower_full[:, -horizon:]
        upper = raw_upper_full[:, -horizon:]
        width = upper - lower
        return ConformalInterval(lower=lower, upper=upper, width=width)

    @property
    def quantile(self) -> Optional[torch.Tensor]:
        """Return the learned calibration quantile."""

        return self._quantile


__all__ = ["BlockConformalCalibrator", "ConformalInterval"]
