"""Minimum-description-length penalties for segmented warps."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from ..transforms.time_warp import MonotoneTimeWarp
from ..transforms.value_transform import MonotoneRQTransform


@dataclass
class MDLPenaltyConfig:
    zeta_1: float = 1.0
    zeta_2: float = 0.5
    zeta_3: float = 0.25


def mdl_penalty(
    warp: MonotoneTimeWarp,
    value_transform: MonotoneRQTransform,
    length: int,
    *,
    config: MDLPenaltyConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Return MDL-style penalties for the segmentation complexity."""

    if length <= 1:
        raise ValueError("sequence length must be greater than one")
    coeffs = config or MDLPenaltyConfig()
    num_segments = max(int(warp.anchors.numel()) - 1, 1)
    knots_per_segment: Sequence[int] = [value_transform.n_bins] * num_segments
    log_length = math.log(max(length, 2))
    loglog_length = math.log(max(math.log(max(length, 3)), 1.0) + 1.0)
    mdl_value = (
        coeffs.zeta_1 * num_segments * log_length
        + coeffs.zeta_2 * sum(knots_per_segment) * log_length
        + coeffs.zeta_3 * num_segments * loglog_length
    )
    complexity = warp.complexity()
    device = warp.anchors.device
    penalty = {
        "mdl_structure": torch.tensor(mdl_value, device=device),
        "mdl_curvature": complexity.get("curvature", torch.tensor(0.0, device=device)),
        "mdl_log_derivative": complexity.get(
            "log_derivative", torch.tensor(0.0, device=device)
        ),
    }
    return penalty


__all__ = ["MDLPenaltyConfig", "mdl_penalty"]
