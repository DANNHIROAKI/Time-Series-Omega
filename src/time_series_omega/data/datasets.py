"""Dataset abstractions for SFF-Ω experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceBatch:
    values: torch.Tensor
    covariates: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


class InMemorySequenceDataset(Dataset[SequenceBatch]):
    """Simple dataset that keeps sequences in memory.

    Parameters
    ----------
    values:
        Array of shape ``(num_series, length, channels)``.
    covariates:
        Optional array of exogenous variables with matching first two dims.
    mask:
        Optional boolean mask for missing data; ``True`` indicates observed.
    device:
        Target device; defaults to CPU. Values are converted to float32.
    """

    def __init__(
        self,
        values: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if values.ndim != 3:
            raise ValueError("values must be (num_series, length, channels)")
        self.values = torch.as_tensor(values, dtype=torch.float32, device=device)
        self.covariates = (
            torch.as_tensor(covariates, dtype=torch.float32, device=device)
            if covariates is not None
            else None
        )
        self.mask = (
            torch.as_tensor(mask, dtype=torch.bool, device=device)
            if mask is not None
            else None
        )

    def __len__(self) -> int:  # type: ignore[override]
        return self.values.shape[0]

    def __getitem__(self, idx: int) -> SequenceBatch:  # type: ignore[override]
        values = self.values[idx]
        covariates = None if self.covariates is None else self.covariates[idx]
        mask = None if self.mask is None else self.mask[idx]
        return SequenceBatch(values, covariates, mask)


def collate_batches(batch: Sequence[SequenceBatch]) -> SequenceBatch:
    values = torch.stack([item.values for item in batch], dim=0)
    covariates = None
    mask = None
    if batch[0].covariates is not None:
        covariates = torch.stack([item.covariates for item in batch], dim=0)
    if batch[0].mask is not None:
        mask = torch.stack([item.mask for item in batch], dim=0)
    return SequenceBatch(values=values, covariates=covariates, mask=mask)


def rolling_windows(
    array: np.ndarray,
    window: int,
    horizon: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling input/target pairs."""
    if array.ndim != 2:
        raise ValueError("array must be 2D (length, channels)")
    inputs = []
    targets = []
    for start in range(0, array.shape[0] - window - horizon + 1, step):
        end = start + window
        inputs.append(array[start:end])
        targets.append(array[end : end + horizon])
    return np.stack(inputs), np.stack(targets)


__all__ = ["SequenceBatch", "InMemorySequenceDataset", "collate_batches", "rolling_windows"]
