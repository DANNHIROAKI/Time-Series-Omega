"""Command line interface for training the SFF-Ω model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from time_series_omega.coverage import BlockConformalCalibrator
from time_series_omega.data.datasets import InMemorySequenceDataset, collate_batches
from time_series_omega.models.sff_omega import SFFOmega
from time_series_omega.training.pipeline import AdversarialConfig, TrainConfig, Trainer


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
    parser.add_argument("--adv", action="store_true", help="Enable adversarial training")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def synthetic_dataset(num_series: int, length: int, channels: int) -> InMemorySequenceDataset:
    t = np.linspace(0, 4 * np.pi, length)
    data = []
    covariates = []
    for _ in range(num_series):
        phase = np.random.uniform(0, np.pi)
        freq = np.random.uniform(0.9, 1.1)
        signal = np.stack([np.sin(freq * t + phase) for _ in range(channels)], axis=-1)
        data.append(signal)
        cov = np.stack([np.sin(t), np.cos(t)], axis=-1)
        covariates.append(cov)
    array = np.stack(data)
    cov_array = np.stack(covariates)
    return InMemorySequenceDataset(array, covariates=cov_array)


def main() -> None:
    args = parse_args()
    dataset = synthetic_dataset(32, args.length + args.horizon, args.channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batches)
    model = SFFOmega(
        length=args.length + args.horizon,
        channels=args.channels,
        horizon=args.horizon,
        calendar_features=["sin_time", "cos_time"],
        calendar_period=args.length,
    )
    adversarial = AdversarialConfig(enabled=args.adv)
    config = TrainConfig(
        epochs=args.epochs,
        device=args.device,
        consensus_weight=args.consensus_weight,
        adversarial=adversarial,
    )
    trainer = Trainer(model, config)
    trainer.fit(loader)
    metrics = trainer.evaluate(loader)
    print("Training reconstruction metrics:", metrics)
    residuals, mask = trainer.canonical_residuals(loader)
    calibrator = BlockConformalCalibrator(model, block_length=16, alpha=0.1)
    calibrator.fit(residuals, mask=mask)
    sample = dataset[0]
    sample_values = sample.values.unsqueeze(0).to(args.device)
    sample_covariates = (
        None if sample.covariates is None else sample.covariates.unsqueeze(0).to(args.device)
    )
    sample_mask = None if sample.mask is None else sample.mask.unsqueeze(0).to(args.device)
    _, gauge = model(sample_values, covariates=sample_covariates, mask=sample_mask, return_gauge=True)  # type: ignore[arg-type]
    interval = calibrator.calibrate(gauge)
    print("Example conformal interval width:", interval.width.mean().item())


if __name__ == "__main__":
    main()
