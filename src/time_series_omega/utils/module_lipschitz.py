"""Utilities for enforcing Lipschitz constraints on PyTorch modules."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def spectral_normalize(module: nn.Module, n_power_iterations: int = 1) -> nn.Module:
    """Apply spectral normalization to all linear/conv layers in ``module``.

    Spectral normalization is used as a lightweight mechanism to control the
    Lipschitz constant of the residual network described in the SFF-Ω framework.
    The helper walks the module tree and wraps supported layers in
    ``torch.nn.utils.spectral_norm`` if they have not already been normalized.
    """

    for child_name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):  # extend as needed
            if not hasattr(child, "weight_orig"):
                nn.utils.spectral_norm(child, n_power_iterations=n_power_iterations)
        else:
            spectral_normalize(child, n_power_iterations=n_power_iterations)
    return module


def lipschitz_penalty(parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Quadratic penalty that activates when the gradient norm exceeds ``max_norm``."""
    total = torch.tensor(0.0, device=next(iter(parameters)).device)
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = param.grad.norm()
        if grad_norm > max_norm:
            total = total + (grad_norm - max_norm) ** 2
    return total


class LipschitzClipper:
    """Gradient clipper that enforces a global Lipschitz threshold."""

    def __init__(self, max_norm: float) -> None:
        self.max_norm = max_norm

    def __call__(self, module: nn.Module) -> None:
        for param in module.parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.norm()
            if grad_norm > self.max_norm:
                param.grad.mul_(self.max_norm / (grad_norm + 1e-12))


__all__ = ["spectral_normalize", "lipschitz_penalty", "LipschitzClipper"]
