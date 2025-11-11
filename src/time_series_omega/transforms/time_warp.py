"""Time reparameterisation modules."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MonotoneTimeWarp(nn.Module):
    """Differentiable monotone reparameterisation of the time axis.

    The module parametrises a strictly positive velocity profile over a unit
    interval, integrates it to obtain a monotonically increasing sampling grid
    and finally rescales the grid to the desired sequence length.  The
    resulting warp can be applied to batched sequences through differentiable
    grid sampling which enables gradients to flow back to the warp parameters.
    """

    def __init__(
        self,
        length: int,
        min_speed: float = 1e-2,
        max_compression: float = 10.0,
        smoothing: float = 1e-3,
    ) -> None:
        super().__init__()
        if length < 2:
            raise ValueError("length must be >= 2")
        self.length = length
        self.min_speed = float(min_speed)
        self.max_compression = float(max_compression)
        self.smoothing = float(smoothing)
        # Raw velocities are unconstrained parameters that will be mapped to a
        # strictly positive profile by ``softplus``.
        self.raw_velocity = nn.Parameter(torch.zeros(length))

    def forward(self) -> torch.Tensor:  # type: ignore[override]
        """Return the cumulative warp grid with values in ``[0, length - 1]``."""

        velocity = torch.nn.functional.softplus(self.raw_velocity) + self.min_speed
        velocity = velocity.clamp(max=self.max_compression)
        # Normalise the mean velocity so that the end point is exactly aligned
        # with ``length - 1``.  Clamping avoids degenerate compression.
        velocity = velocity / velocity.mean().clamp_min(1e-6)
        cumulative = torch.cumsum(velocity, dim=-1)
        cumulative = cumulative / cumulative[-1]
        grid = cumulative * (self.length - 1)
        return grid

    def _second_difference(self, values: torch.Tensor) -> torch.Tensor:
        pad = torch.nn.functional.pad(values, (1, 1), mode="replicate")
        return pad[:-2] - 2 * pad[1:-1] + pad[2:]

    def curvature_penalty(self) -> torch.Tensor:
        """Squared ``H^2`` semi-norm promoting smooth warps."""

        grid = self.forward()
        second_diff = self._second_difference(grid)
        return torch.mean(second_diff**2)

    def log_derivative_penalty(self) -> torch.Tensor:
        """Penalty on the log-derivative variation along the grid."""

        velocity = torch.nn.functional.softplus(self.raw_velocity) + self.min_speed
        log_velocity = torch.log(velocity)
        centred = log_velocity - log_velocity.mean()
        diffs = centred[1:] - centred[:-1]
        return torch.mean(diffs**2)

    def apply(self, values: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
        """Warp ``values`` along the time axis.

        Parameters
        ----------
        values:
            Tensor of shape ``(..., length, channels)``.
        align_corners:
            Passed to :func:`torch.nn.functional.grid_sample`.
        """

        if values.shape[-2] != self.length:
            raise ValueError(
                f"expected time dimension {self.length}, received {values.shape[-2]}"
            )
        grid = self.forward()
        # Grid-sample expects coordinates in [-1, 1].
        normalised = (grid / (self.length - 1) * 2 - 1).view(*([1] * (values.dim() - 2)), self.length, 1)
        # Prepare grid of shape (batch, length, 1, 2) for 1D sampling.
        view_shape = (*values.shape[:-2], self.length, 1)
        base = normalised.expand(view_shape)
        grid_1d = torch.stack((base, torch.zeros_like(base)), dim=-1)
        flat = values.reshape(-1, self.length, values.shape[-1]).transpose(1, 2).unsqueeze(-1)
        warped = F.grid_sample(flat, grid_1d.reshape(-1, self.length, 1, 2), align_corners=align_corners)
        warped = warped.squeeze(-1).transpose(1, 2)
        warped = warped.reshape(*values.shape)
        return warped


__all__ = ["MonotoneTimeWarp"]
