"""Command line interface for training the SFF-Ω model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from time_series_omega.data.datasets import InMemorySequenceDataset, SequenceBatch
from time_series_omega.models.sff_omega import SFFOmega
from time_series_omega.training.pipeline import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SFF-Ω model on a toy dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--memory", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--consensus-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def synthetic_dataset(num_series: int, length: int, channels: int) -> InMemorySequenceDataset:
    t = np.linspace(0, 4 * np.pi, length)
    data = []
    for _ in range(num_series):
        phase = np.random.uniform(0, np.pi)
        freq = np.random.uniform(0.9, 1.1)
        signal = np.stack([np.sin(freq * t + phase) for _ in range(channels)], axis=-1)
        data.append(signal)
    array = np.stack(data)
    return InMemorySequenceDataset(array)


def main() -> None:
    args = parse_args()
    dataset = synthetic_dataset(32, args.length + args.horizon, args.channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SFFOmega(length=args.length + args.horizon, channels=args.channels, horizon=args.horizon)
    config = TrainConfig(
        epochs=args.epochs,
        device=args.device,
        consensus_weight=args.consensus_weight,
    )
    trainer = Trainer(model, config)
    trainer.fit(loader)
    metrics = trainer.evaluate(loader)
    print("Training reconstruction metrics:", metrics)


if __name__ == "__main__":
    main()
