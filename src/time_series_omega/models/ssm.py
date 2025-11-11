"""Stable state space model building blocks."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class StableSSM(nn.Module):
    """Linear time-invariant state space with stability constraint."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        memory: int = 1,
        rho: float = 0.95,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory = memory
        self.rho = rho
        self.A_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(memory, hidden_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)
        self.D = nn.Parameter(torch.randn(memory, output_dim, input_dim) * 0.01)

    def _stable_A(self) -> torch.Tensor:
        return torch.tanh(self.A_raw) * self.rho

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch, length, _ = inputs.shape
        A = self._stable_A()
        state = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        outputs = []
        padded = torch.nn.functional.pad(inputs, (0, 0, self.memory - 1, 0))
        for t in range(length):
            past = []
            for lag in range(self.memory):
                past.append(
                    torch.matmul(padded[:, t + self.memory - 1 - lag], self.B[lag].transpose(0, 1))
                )
            control = torch.stack(past, dim=0).sum(dim=0)
            state = torch.matmul(state, A.t()) + control
            y = torch.matmul(state, self.C.t())
            for lag in range(self.memory):
                y = y + torch.matmul(
                    padded[:, t + self.memory - 1 - lag], self.D[lag].transpose(0, 1)
                )
            outputs.append(y)
        return torch.stack(outputs, dim=1)


__all__ = ["StableSSM"]
