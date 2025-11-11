"""Top-level package for the SFF-Ω reference implementation."""

from .models.sff_omega import SFFOmega
from .training.pipeline import DeploymentGate, GateDecision, TrainConfig, Trainer

__all__ = ["SFFOmega", "TrainConfig", "Trainer", "DeploymentGate", "GateDecision"]
