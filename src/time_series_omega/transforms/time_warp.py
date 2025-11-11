"""Time reparameterisation modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class WarpDiagnostics:
    """Container with interpretable statistics of the learned warp."""

    components: Dict[str, torch.Tensor]
    velocity: torch.Tensor
    anchors: Optional[torch.Tensor]


class MonotoneTimeWarp(nn.Module):
    """Differentiable, hierarchical monotone reparameterisation of the time axis.

    The implementation follows the modelling guidelines in the SFF-Ω
    methodology: the warp is represented as the composition of a handful of
    slowly varying factors (``slow``/``fast``/``event``/``local``) whose product
    yields a strictly positive velocity profile.  The profile is normalised to
    preserve the end-point anchors and then integrated to form an increasing
    sampling grid.  Optional segment anchors can be supplied to guarantee that
    the warp keeps prescribed breakpoints fixed, which is necessary for the
    MDL-style segmentation procedure described in the paper.
    """

    DEFAULT_COMPONENTS: Tuple[str, ...] = ("slow", "fast", "event", "local")

    def __init__(
        self,
        length: int,
        *,
        components: Iterable[str] = DEFAULT_COMPONENTS,
        segments: Optional[Iterable[int]] = None,
        min_speed: float = 1e-3,
        max_compression: float = 16.0,
        smoothing: float = 1e-2,
    ) -> None:
        super().__init__()
        if length < 2:
            raise ValueError("length must be >= 2")
        self.length = int(length)
        self.min_speed = float(min_speed)
        self.max_compression = float(max_compression)
        self.smoothing = float(smoothing)
        self.components = tuple(dict.fromkeys(components)) or self.DEFAULT_COMPONENTS
        params: Dict[str, nn.Parameter] = {}
        for name in self.components:
            params[name] = nn.Parameter(torch.zeros(length))
            self.register_parameter(f"raw_{name}", params[name])
        if segments is None:
            anchors = torch.tensor([0, length - 1], dtype=torch.long)
        else:
            anchor_list: List[int] = sorted(set(int(i) for i in segments if 0 <= i < length))
            if 0 not in anchor_list:
                anchor_list.insert(0, 0)
            if length - 1 not in anchor_list:
                anchor_list.append(length - 1)
            anchors = torch.tensor(anchor_list, dtype=torch.long)
        self.register_buffer("anchors", anchors, persistent=False)

    # ------------------------------------------------------------------
    # Velocity construction utilities
    # ------------------------------------------------------------------
    def _component_velocity(self, name: str) -> torch.Tensor:
        raw = getattr(self, f"raw_{name}")
        velocity = torch.nn.functional.softplus(raw) + self.min_speed
        if name in {"slow", "fast"}:
            kernel = torch.ones(1, 1, 7 if name == "slow" else 3, device=velocity.device)
            kernel = kernel / kernel.sum()
            velocity = velocity.view(1, 1, -1)
            velocity = torch.nn.functional.conv1d(velocity, kernel, padding=kernel.shape[-1] // 2)
            velocity = velocity.view(-1)
        if name == "event":
            sigma = max(self.smoothing, 1e-3)
            radius = 2
            positions = torch.arange(-radius, radius + 1, device=velocity.device, dtype=velocity.dtype)
            kernel = torch.exp(-(positions**2) / (2 * sigma**2))
            kernel = (kernel / kernel.sum()).view(1, 1, -1)
            velocity = velocity.view(1, 1, -1)
            velocity = torch.nn.functional.conv1d(
                velocity,
                kernel,
                padding=radius,
            )
            velocity = velocity.view(-1)
        return velocity.clamp_max(self.max_compression)

    def velocity(self, normalise: bool = True) -> torch.Tensor:
        """Return the combined velocity profile.

        Parameters
        ----------
        normalise:
            If ``True`` (default) the mean velocity is normalised to one so
            that the integrated grid ends exactly at ``length - 1``.
        """

        prod = torch.ones(self.length, device=self.anchors.device)
        for name in self.components:
            prod = prod * self._component_velocity(name)
        prod = prod.clamp_min(self.min_speed)
        if normalise:
            prod = prod / prod.mean().clamp_min(1e-6)
        return prod

    # ------------------------------------------------------------------
    # Grid construction and penalties
    # ------------------------------------------------------------------
    def forward(self, return_diagnostics: bool = False) -> torch.Tensor | Tuple[torch.Tensor, WarpDiagnostics]:  # type: ignore[override]
        velocity = self.velocity()
        cumulative = torch.cumsum(velocity, dim=-1)
        cumulative = cumulative / cumulative[-1]
        grid = cumulative * (self.length - 1)

        if self.anchors.numel() > 1:
            grid = grid.clone()
            for idx in range(self.anchors.numel() - 1):
                start = int(self.anchors[idx].item())
                end = int(self.anchors[idx + 1].item())
                if end <= start:
                    continue
                anchor_segment = torch.linspace(
                    grid[start],
                    grid[end],
                    steps=end - start + 1,
                    device=grid.device,
                )
                grid[start : end + 1] = anchor_segment

        if not return_diagnostics:
            return grid

        components = {name: self._component_velocity(name) for name in self.components}
        diagnostics = WarpDiagnostics(components=components, velocity=velocity, anchors=self.anchors)
        return grid, diagnostics

    def _second_difference(self, values: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(
            values.unsqueeze(0).unsqueeze(0),
            (1, 1),
            mode="replicate",
        ).squeeze(0).squeeze(0)
        return padded[:-2] - 2 * padded[1:-1] + padded[2:]

    def curvature_penalty(self) -> torch.Tensor:
        grid = self.forward(return_diagnostics=False)  # type: ignore[arg-type]
        second_diff = self._second_difference(grid)
        return torch.mean(second_diff**2)

    def log_derivative_penalty(self) -> torch.Tensor:
        velocity = self.velocity(normalise=False)
        log_velocity = torch.log(velocity)
        centred = log_velocity - log_velocity.mean()
        diffs = centred[1:] - centred[:-1]
        return torch.mean(diffs**2)

    def complexity(self) -> Dict[str, torch.Tensor]:
        """Return individual complexity terms useful for MDL penalties."""

        terms = {"curvature": self.curvature_penalty(), "log_derivative": self.log_derivative_penalty()}
        for name in self.components:
            component = self._component_velocity(name)
            terms[f"{name}_energy"] = component.mean()
        return terms

    # ------------------------------------------------------------------
    # Application to sequences
    # ------------------------------------------------------------------
    def apply(
        self,
        values: torch.Tensor,
        *,
        align_corners: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Warp ``values`` along the time axis."""

        if values.shape[-2] != self.length:
            raise ValueError(
                f"expected time dimension {self.length}, received {values.shape[-2]}"
            )
        grid = self.forward(return_diagnostics=False)  # type: ignore[arg-type]
        normalised = (grid / (self.length - 1) * 2 - 1).view(*([1] * (values.dim() - 2)), self.length, 1)
        view_shape = (*values.shape[:-2], self.length, 1)
        base = normalised.expand(view_shape)
        grid_1d = torch.stack((base, torch.zeros_like(base)), dim=-1)
        flat = values.reshape(-1, self.length, values.shape[-1]).transpose(1, 2).unsqueeze(-1)
        warped = F.grid_sample(
            flat,
            grid_1d.reshape(-1, self.length, 1, 2),
            align_corners=align_corners,
        )
        warped = warped.squeeze(-1).transpose(1, 2)
        warped = warped.reshape(*values.shape)
        if mask is not None:
            warped = warped * mask.to(warped.dtype)
        return warped

    def inverse_apply(
        self,
        values: torch.Tensor,
        *,
        align_corners: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Approximate inverse warp by interpolating onto the raw timeline."""

        if values.shape[-2] != self.length:
            raise ValueError(
                f"expected time dimension {self.length}, received {values.shape[-2]}"
            )
        grid = self.forward(return_diagnostics=False)  # type: ignore[arg-type]
        canonical = torch.linspace(0.0, self.length - 1.0, self.length, device=grid.device)
        bucket = torch.bucketize(canonical, grid)
        left_idx = (bucket - 1).clamp(min=0, max=self.length - 1)
        right_idx = bucket.clamp(min=0, max=self.length - 1)
        grid_left = grid[left_idx]
        grid_right = grid[right_idx]
        span = (grid_right - grid_left).clamp_min(1e-6)
        alpha = (canonical - grid_left) / span
        raw_positions = left_idx.to(values.dtype) + alpha * (right_idx - left_idx).to(values.dtype)
        normalised = (raw_positions / (self.length - 1) * 2 - 1).view(
            *([1] * (values.dim() - 2)), self.length, 1
        )
        view_shape = (*values.shape[:-2], self.length, 1)
        base = normalised.expand(view_shape)
        grid_1d = torch.stack((base, torch.zeros_like(base)), dim=-1)
        flat = values.reshape(-1, self.length, values.shape[-1]).transpose(1, 2).unsqueeze(-1)
        restored = F.grid_sample(
            flat,
            grid_1d.reshape(-1, self.length, 1, 2),
            align_corners=align_corners,
        )
        restored = restored.squeeze(-1).transpose(1, 2).reshape(*values.shape)
        if mask is not None:
            restored = restored * mask.to(restored.dtype)
        return restored


__all__ = ["MonotoneTimeWarp", "WarpDiagnostics"]
