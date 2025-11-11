# Time-Series-Omega

This repository provides a reference implementation of the **SFF-Ω** framework
outlined in the accompanying research notes. The implementation is written in
PyTorch and organises the workflow into explicit *gauge* components, a stable
state space backbone and a small Lipschitz residual module. The code base is
structured so that each part of the theoretical pipeline can be inspected and
extended independently.

## Features

- Hierarchical monotone time reparameterisation with curvature, log-slope and
  component-wise complexity control together with optional segmentation anchors.
- Channel-aware monotone rational–quadratic value-domain transform with linear
  tails and log-Jacobian support.
- Learnable calendar alignment flow combining soft permutations, affine
  normalisation and differentiable seasonal phase shifts.
- Sliding-window moment stabilisation, STFT-based soft anchors, pairwise
  consensus energy and H-disc low-pass checks for canonical alignment.
- Stable linear state space model with learnable control kernels.
- Spectrally normalised residual MLP enforcing small Lipschitz constants.
- Training utilities with configurable regularisation weights, adversarial
  diffeomorphism perturbations, consensus gates and gradient clipping.
- Validation helpers for MAE/RMSE assessment and a risk-difference gate based on
  statistical predictive intervals.
- Simple synthetic data generator to exercise the full pipeline end-to-end.

## Installation

The project targets Python 3.10+ and depends on PyTorch. After creating and
activating a virtual environment, install the package in editable mode:

```bash
pip install -e .
```

If you prefer not to install the package, set `PYTHONPATH=src` when running the
scripts.

## Quickstart

To train the SFF-Ω model on the built-in sinusoidal toy dataset run:

```bash
python scripts/train.py --epochs 10 --length 128 --horizon 24 --channels 1 --adv
```

Command line arguments allow you to adjust the temporal context, forecast
horizon, batch size, device and whether adversarial training should be enabled.

## Project Layout

```
src/time_series_omega/
  data/              # dataset abstractions and rolling window helpers
  transforms/        # gauge components: time warp, value transform, calendar
  losses/            # canonical domain regularisers
  models/            # stable SSM, Lipschitz residual and wrapper module
  training/          # trainer and configuration dataclasses
scripts/
  train.py           # example training entry point
```

Each module contains docstrings describing its responsibilities and key design
choices. The implementation is intentionally modular to simplify replacement or
extension of individual components (e.g. swapping in a different state-space
architecture, or experimenting with alternate gauge constraints).

## Extending the Framework

The code base now includes differentiable calendar flows, adversarial diffeo
training, consensus-aware regularisation and low-pass anti-alias checks. The
architecture remains modular so that researchers can plug in alternate state
space blocks, conformal calibration heads or segmentation objectives with
minimal friction. Contributions and pull requests are welcome.

