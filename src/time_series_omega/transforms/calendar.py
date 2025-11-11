"""Calendar alignment utilities."""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class CalendarAlignment(nn.Module):
    """Learnable, invertible alignment of calendar features.

    The alignment acts as a small normalising flow over exogenous variables:
    - a soft permutation matrix learns to reorder features in a nearly
      bijective fashion;
    - per-feature affine parameters capture scale/shift differences;
    - optional periodic phase shifts enable soft rolling of seasonal signals.

    The transformation is fully differentiable and admits an efficient inverse
    used when mapping predictions back to the original feature space.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        period: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.index: Dict[str, int] = {name: i for i, name in enumerate(self.feature_names)}
        self.temperature = float(temperature)
        n_features = len(self.feature_names)
        if n_features == 0:
            raise ValueError("CalendarAlignment requires at least one feature")
        self.logits = nn.Parameter(torch.eye(n_features))
        self.shift = nn.Parameter(torch.zeros(n_features))
        self.log_scale = nn.Parameter(torch.zeros(n_features))
        self.period = period
        if period is not None:
            self.phase = nn.Parameter(torch.zeros(n_features))
        else:
            self.register_parameter("phase", None)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def permutation_matrix(self) -> torch.Tensor:
        weights = self.logits / max(self.temperature, 1e-3)
        return F.softmax(weights, dim=-1)

    def regulariser(self) -> torch.Tensor:
        """Return a penalty encouraging nearly permutation matrices."""

        perm = self.permutation_matrix()
        gram = perm @ perm.transpose(0, 1)
        identity = torch.eye(perm.shape[0], device=perm.device)
        return torch.mean((gram - identity) ** 2)

    # ------------------------------------------------------------------
    # Forward / inverse transformations
    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        perm = self.permutation_matrix()
        aligned = torch.tensordot(features, perm.transpose(0, 1), dims=([-1], [0]))
        aligned = aligned * torch.exp(self.log_scale) + self.shift
        if self.period is not None and self.phase is not None and aligned.shape[-2] >= self.period:
            base = aligned[..., : self.period, :]
            freq = torch.fft.rfft(base, dim=-2)
            k = torch.arange(freq.shape[-2], device=aligned.device, dtype=aligned.dtype)
            phase = torch.tanh(self.phase) * math.pi
            phase = phase.view(*([1] * (freq.dim() - 1)), -1)
            exponent = -1j * k.view(*([1] * (freq.dim() - 2)), -1, 1) * phase
            shift_factor = torch.exp(exponent)
            freq = freq * shift_factor
            shifted = torch.fft.irfft(freq, n=self.period, dim=-2)
            aligned = aligned.clone()
            aligned[..., : self.period, :] = shifted
        return aligned

    def inverse(self, features: torch.Tensor) -> torch.Tensor:
        perm = self.permutation_matrix()
        scale = torch.exp(self.log_scale).clamp_min(1e-6)
        inv = (features - self.shift) / scale
        inv = torch.tensordot(inv, perm, dims=([-1], [1]))
        return inv


__all__ = ["CalendarAlignment"]
