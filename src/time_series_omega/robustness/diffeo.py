"""Diffeomorphic adversarial perturbations used in the training pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..losses.regularizers import low_pass_filter


def _apply_diffeomorphism(
    series: torch.Tensor,
    delta: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Warp ``series`` by the discrete diffeomorphism ``Id + delta``.

    Parameters
    ----------
    series:
        Tensor of shape ``(batch, length, channels)``.
    delta:
        Tensor of shape ``(batch, length, 1)`` with zero boundary conditions.
    mask:
        Optional mask; entries that are ``False`` are set to zero after warping.
    """

    if delta.shape[:-1] != series.shape[:-1] or delta.shape[-1] != 1:
        raise ValueError("delta must have shape (batch, length, 1)")
    batch, length, channels = series.shape
    base = torch.linspace(0.0, length - 1.0, length, device=series.device, dtype=series.dtype)
    base = base.view(1, length, 1)
    warped_grid = base + delta
    warped_grid = torch.clamp(warped_grid, 0.0, length - 1.0)
    normalised = warped_grid / (length - 1.0) * 2.0 - 1.0
    grid = torch.stack((normalised, torch.zeros_like(normalised)), dim=-1)
    flat = series.reshape(batch, length, channels).transpose(1, 2).unsqueeze(-1)
    warped = nn.functional.grid_sample(
        flat,
        grid.reshape(batch, length, 1, 2),
        align_corners=True,
    )
    warped = warped.squeeze(-1).transpose(1, 2)
    warped = warped.reshape(batch, length, channels)
    if mask is not None:
        warped = warped * mask.to(warped.dtype)
    return warped


@dataclass
class DiffeomorphismConstraints:
    epsilon_inf: float = 0.05
    epsilon_log: float = 0.1


class DiffeomorphicAdversary:
    """Projected-gradient adversary constrained to small diffeomorphisms."""

    def __init__(
        self,
        *,
        steps: int = 3,
        step_size: float = 0.02,
        cutoff: float = 0.45,
        constraints: Optional[DiffeomorphismConstraints] = None,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.steps = steps
        self.step_size = step_size
        self.cutoff = cutoff
        self.constraints = constraints or DiffeomorphismConstraints()

    # ------------------------------------------------------------------
    # Projection utilities
    # ------------------------------------------------------------------
    def _project(self, delta: torch.Tensor) -> torch.Tensor:
        """Project ``delta`` onto the admissible diffeomorphism set."""

        eps = self.constraints.epsilon_inf
        eps_log = self.constraints.epsilon_log
        # Enforce boundary conditions
        delta = delta.clone()
        delta[:, 0] = 0.0
        delta[:, -1] = 0.0
        # Clamp the discrete derivative
        diff = delta[:, 1:] - delta[:, :-1]
        diff = diff.clamp(min=-eps, max=eps)
        # Reconstruct delta with zero endpoints
        recon = torch.zeros_like(delta)
        recon[:, 1:] = torch.cumsum(diff, dim=1)
        # Remove any accumulated drift so that the final point is zero
        length = delta.shape[1]
        indices = torch.linspace(0.0, 1.0, length, device=delta.device, dtype=delta.dtype)
        drift = recon[:, -1:, :]
        recon = recon - indices.view(1, -1, 1) * drift
        recon[:, 0] = 0.0
        recon[:, -1] = 0.0
        # Control the log-derivative energy by scaling when necessary
        diff = recon[:, 1:] - recon[:, :-1]
        safe_diff = diff.clamp(min=-0.95, max=0.95)
        log_velocity = torch.log1p(safe_diff)
        grad_log = log_velocity[:, 1:] - log_velocity[:, :-1]
        energy = torch.linalg.norm(grad_log.reshape(grad_log.shape[0], -1), dim=1)
        limit = eps_log * torch.sqrt(torch.tensor(diff.shape[1], device=delta.device, dtype=delta.dtype))
        mask = energy > limit
        if mask.any():
            scale = (limit[mask] / energy[mask]).view(-1, 1, 1).clamp(max=1.0)
            recon[mask] = recon[mask] * scale
        return recon

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def perturb(
        self,
        model: nn.Module,
        series: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return an adversarially warped version of ``series``."""

        device = series.device
        filtered = low_pass_filter(series, cutoff=self.cutoff)
        residual = series - filtered
        batch, length, _ = series.shape
        delta = torch.zeros(batch, length, 1, device=device, dtype=series.dtype, requires_grad=True)
        target_tensor = target if target is not None else series[:, -model.horizon :]
        for _ in range(self.steps):
            warped = _apply_diffeomorphism(filtered, delta, mask=mask)
            adversarial = warped + residual
            pred = model(adversarial, mask=mask)
            loss = nn.functional.mse_loss(pred, target_tensor)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            step = self.step_size * torch.sign(grad)
            delta = (delta + step).detach()
            delta = self._project(delta)
            delta.requires_grad_(True)
        warped = _apply_diffeomorphism(filtered, delta, mask=mask)
        return warped + residual


__all__ = ["DiffeomorphicAdversary", "DiffeomorphismConstraints"]
