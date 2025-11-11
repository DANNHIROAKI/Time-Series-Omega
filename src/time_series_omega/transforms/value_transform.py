"""Value-domain monotone transformations used by SFF-Ω."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _normalise_params(
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    widths = F.softmax(widths, dim=-1)
    widths = widths * (1 - min_bin_width * widths.shape[-1]) + min_bin_width
    heights = F.softmax(heights, dim=-1)
    heights = heights * (1 - min_bin_height * heights.shape[-1]) + min_bin_height
    derivatives = F.softplus(derivatives) + min_derivative
    return widths, heights, derivatives


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Return searchsorted indices matching the trailing channel shape."""

    mask = inputs[..., None] >= bin_locations[..., :-1]
    return mask.sum(dim=-1)


def _gather_at(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_dim = params.dim() - 1
    expanded = indices.unsqueeze(-1)
    return torch.gather(params, gather_dim, expanded).squeeze(-1)


def _rational_quadratic_spline(
    inputs: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    inverse: bool,
    tail_bound: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_shape = inputs.shape[:-1]
    channels = inputs.shape[-1]
    flat_inputs = inputs.reshape(-1, channels)

    widths = widths.unsqueeze(0).expand(flat_inputs.shape[0], -1, -1)
    heights = heights.unsqueeze(0).expand_as(widths)
    derivatives = derivatives.unsqueeze(0).expand(flat_inputs.shape[0], -1, -1)

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)

    cumwidths = cumwidths / cumwidths[..., -1:].clamp_min(eps)
    cumheights = cumheights / cumheights[..., -1:].clamp_min(eps)

    cumwidths = 2 * tail_bound * cumwidths - tail_bound
    cumheights = 2 * tail_bound * cumheights - tail_bound

    inputs_clamped = flat_inputs.clamp(-tail_bound + eps, tail_bound - eps)
    bin_idx = _searchsorted(cumwidths, inputs_clamped) - 1
    bin_idx = bin_idx.clamp(min=0, max=widths.shape[-1] - 1)
    next_idx = (bin_idx + 1).clamp(max=widths.shape[-1])

    input_cumwidths = _gather_at(cumwidths, bin_idx)
    input_upper_cumwidths = _gather_at(cumwidths, next_idx)
    input_bin_widths = input_upper_cumwidths - input_cumwidths

    input_cumheights = _gather_at(cumheights, bin_idx)
    input_upper_cumheights = _gather_at(cumheights, next_idx)
    input_bin_heights = input_upper_cumheights - input_cumheights

    derivative_left = _gather_at(derivatives, bin_idx)
    derivative_right = _gather_at(derivatives, next_idx)

    delta = input_bin_heights / input_bin_widths

    if inverse:
        target = inputs_clamped
        theta = ((target - input_cumheights) / input_bin_heights).clamp(0.0, 1.0)
        for _ in range(6):
            numerator = input_bin_heights * (
                delta * theta**2 + derivative_left * theta * (1 - theta)
            )
            denominator = delta + (derivative_left + derivative_right - 2 * delta) * theta * (1 - theta)
            y_theta = input_cumheights + numerator / denominator
            derivative_numerator = delta**2 * (
                derivative_right * theta**2
                + 2 * delta * theta * (1 - theta)
                + derivative_left * (1 - theta) ** 2
            )
            derivative_denominator = denominator**2
            derivative_theta = (derivative_numerator / derivative_denominator).clamp_min(eps)
            theta = (theta - (y_theta - target) / derivative_theta).clamp(0.0, 1.0)
        outputs = input_cumwidths + theta * input_bin_widths
    else:
        theta = ((inputs_clamped - input_cumwidths) / input_bin_widths).clamp(0.0, 1.0)
        numerator = input_bin_heights * (
            delta * theta**2 + derivative_left * theta * (1 - theta)
        )
        denominator = delta + (derivative_left + derivative_right - 2 * delta) * theta * (1 - theta)
        outputs = input_cumheights + numerator / denominator

    numerator = input_bin_heights * (
        delta * theta**2 + derivative_left * theta * (1 - theta)
    )
    denominator = delta + (derivative_left + derivative_right - 2 * delta) * theta * (1 - theta)
    derivative_numerator = delta**2 * (
        derivative_right * theta**2
        + 2 * delta * theta * (1 - theta)
        + derivative_left * (1 - theta) ** 2
    )
    derivative_denominator = denominator**2
    logabsdet = (
        torch.log(derivative_numerator.clamp_min(eps))
        - torch.log(derivative_denominator.clamp_min(eps))
        - torch.log(input_bin_widths.clamp_min(eps))
    )
    if inverse:
        logabsdet = -logabsdet

    outside_mask = torch.logical_or(flat_inputs > tail_bound, flat_inputs < -tail_bound)
    outputs[outside_mask] = flat_inputs[outside_mask]
    logabsdet[outside_mask] = 0.0

    outputs = outputs.reshape(*batch_shape, channels)
    logabsdet = logabsdet.reshape(*batch_shape, channels)
    return outputs, logabsdet


class MonotoneRQTransform(nn.Module):
    """Channel-wise monotone rational-quadratic spline with linear tails."""

    def __init__(
        self,
        channels: int,
        n_bins: int = 8,
        tail_bound: float = 3.0,
        min_bin_width: float = 1e-2,
        min_bin_height: float = 1e-2,
        min_derivative: float = 1e-2,
    ) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.channels = channels
        self.n_bins = n_bins
        self.tail_bound = float(tail_bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)
        self.widths = nn.Parameter(torch.zeros(channels, n_bins))
        self.heights = nn.Parameter(torch.zeros(channels, n_bins))
        self.derivatives = nn.Parameter(torch.zeros(channels, n_bins + 1))

    def _params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _normalise_params(
            self.widths,
            self.heights,
            self.derivatives,
            self.min_bin_width,
            self.min_bin_height,
            self.min_derivative,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        widths, heights, derivatives = self._params()
        outputs, _ = _rational_quadratic_spline(
            inputs,
            widths,
            heights,
            derivatives,
            inverse=False,
            tail_bound=self.tail_bound,
        )
        return outputs

    def inverse(self, inputs: torch.Tensor) -> torch.Tensor:
        widths, heights, derivatives = self._params()
        outputs, _ = _rational_quadratic_spline(
            inputs,
            widths,
            heights,
            derivatives,
            inverse=True,
            tail_bound=self.tail_bound,
        )
        return outputs

    def log_abs_det_jacobian(self, inputs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        widths, heights, derivatives = self._params()
        _, logabsdet = _rational_quadratic_spline(
            inputs,
            widths,
            heights,
            derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )
        return logabsdet


__all__ = ["MonotoneRQTransform"]
