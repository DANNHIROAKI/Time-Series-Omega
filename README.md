# Time-Series-Omega

This repository provides a reference implementation of the **SFF-Ω** framework
outlined in the accompanying research notes. The implementation is written in
PyTorch and organises the workflow into explicit *gauge* components, a stable
state space backbone and a small Lipschitz residual module. The code base is
structured so that each part of the theoretical pipeline can be inspected and
extended independently.

## Features

- Monotone differentiable time reparameterisation with curvature and log-slope
  regularisers.
- Channel-aware monotone rational–quadratic value-domain transform with linear
  tails.
- Sliding-window moment stabilisation, STFT-based soft anchors and pairwise
  consensus energy for canonical alignment.
- Stable linear state space model with learnable control kernels.
- Spectrally normalised residual MLP enforcing small Lipschitz constants.
- Training utilities with configurable regularisation weights, consensus gates
  and gradient clipping.
- Validation helpers for MAE/RMSE assessment and a simple risk-difference gate
  that mirrors the deployment guard described in the paper.
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
python scripts/train.py --epochs 10 --length 128 --horizon 24 --channels 1
```

Command line arguments allow you to adjust the temporal context, forecast
horizon, batch size and device.

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

The current code base ships with a sinusoidal toy generator, yet the
architecture is designed to be extended to multi-variate settings, more complex
calendar alignment schemes, adversarial diffeomorphism robustness objectives and
coverage calibration routines. Contributions and pull requests are welcome.

