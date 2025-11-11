"""Residual Lipschitz network."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..utils.module_lipschitz import spectral_normalize


class LipschitzMLP(nn.Module):
    """Small MLP with spectral normalisation to control Lipschitz constant."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: nn.Module = nn.GELU()) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            linear = nn.Linear(prev, hidden)
            layers.append(linear)
            layers.append(activation)
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        spectral_normalize(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


__all__ = ["LipschitzMLP"]
