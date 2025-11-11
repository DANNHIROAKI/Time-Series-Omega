"""Calendar alignment utilities."""
from __future__ import annotations

from typing import Dict

import torch


class CalendarAlignment:
    """Permutes and rescales calendar features according to learned parameters."""

    def __init__(self, feature_names: Dict[str, int]) -> None:
        self.feature_names = feature_names

    def align(self, features: torch.Tensor, mapping: Dict[str, str]) -> torch.Tensor:
        aligned = torch.zeros_like(features)
        for target, source in mapping.items():
            aligned[..., self.feature_names[target]] = features[..., self.feature_names[source]]
        return aligned


__all__ = ["CalendarAlignment"]
