"""Top-level package for the SFF-Ω reference implementation."""

from .coverage.block_conformal import BlockConformalCalibrator, ConformalInterval
from .models.sff_omega import SFFOmega
from .robustness.diffeo import DiffeomorphicAdversary, DiffeomorphismConstraints
from .training.pipeline import DeploymentGate, GateDecision, TrainConfig, Trainer

__all__ = [
    "SFFOmega",
    "TrainConfig",
    "Trainer",
    "DeploymentGate",
    "GateDecision",
    "DiffeomorphicAdversary",
    "DiffeomorphismConstraints",
    "BlockConformalCalibrator",
    "ConformalInterval",
]
