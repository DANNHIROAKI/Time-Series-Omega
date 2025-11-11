# SFF-Ω Implementation Audit

This document records an internal audit of the repository against the modelling
requirements laid out in the SFF-Ω research specification. The audit was
performed after integrating the changes in this commit.

## Summary

| Pillar | Status | Notes |
| --- | --- | --- |
| Gauge transforms (τ, cal, ν) | **✓ Implemented** | Hierarchical monotone time warp with complexity tracking, calendar alignment flow, and monotone RQ value transform. |
| Canonical regularisers | **✓ Implemented** | Sliding-window stability, STFT soft anchors, consensus energy, H-disc low-pass proxy. |
| Stable dynamics + Lipschitz residual | **✓ Implemented** | Stable SSM core with spectrally normalised residual MLP. |
| Adversarial robustness (Ω-5′) | **△ Partial** | Training loop now applies diffeomorphic PGD with low-pass pre-filtering, but the log-sum-exp surrogate objective is still pending. |
| Coverage calibration (Ω-2′/Ω-3′) | **✓ Implemented** | Canonical-domain block conformal calibrator with β-mixing correction and raw-domain mapping. |
| Segmentation & MDL penalties (Def. 1, Thm. E′) | **△ Partial** | MDL-style penalties are computed from warp anchors/complexity, yet an automatic segmentation search is still absent. |
| Deployment gating (Ω-4′) | **△ Partial** | Validation SPI-style gate implemented; permutation/bootstrap estimators are not yet available. |

## Detailed Findings

1. **Gauge Layer**
   - `MonotoneTimeWarp` now decomposes the time warp into named components and
     reports curvature/log-derivative penalties plus per-component energies.
     Segment anchors guarantee gauge fixing for pre-specified break points.
   - `CalendarAlignment` replaces the placeholder mapping with a differentiable
     normalising-flow-style module (soft permutation, affine scaling, Fourier
     phase shifts) and exposes a regulariser that encourages near-orthogonality.
   - `MonotoneRQTransform` already satisfied the monotone value transform
     requirements.

2. **Canonical Regularisation**
   - The loss suite incorporates sliding-window moment stabilisation, STFT soft
     anchors, pairwise consensus energy and an anti-aliasing low-pass check.
   - Regularisation weights are configurable through `TrainConfig`.

3. **Dynamics**
   - `StableSSM` enforces a contractive state transition matrix while the
     residual MLP is spectrally normalised to keep its Lipschitz constant small.

4. **Robustness & Training Pipeline**
   - `Trainer` now integrates a diffeomorphic PGD adversary built from
     `DiffeomorphicAdversary`, enforcing ε/ε_log constraints after a low-pass
     filtering step as prescribed by H-disc.
   - Training metrics log all regularisers, including calendar, H-disc and MDL
     components. The smooth log-sum-exp surrogate from §2.5 remains to be
     implemented.

5. **Deployment Gate & Coverage**
   - The `DeploymentGate` adopts a statistical predictive interval (SPI) style
     bound based on the empirical variance of loss differences. Additional gates
     (permutation/bootstrap) remain to be implemented.
   - A new `BlockConformalCalibrator` fits block-wise residual quantiles in the
     canonical domain, applies β-mixing corrections and maps calibrated
     intervals back to the raw domain through ν⁻¹ and τ⁻¹, satisfying Ω-2′/Ω-3′.

6. **Missing Components / Future Work**
   - **Segmentation search:** the MDL penalty is present but an optimiser that
     proposes new breakpoints based on the diagnostics is still required.
   - **Template estimation:** external estimation of the soft-anchor template is
     assumed available but not implemented in-code.

## Recommendations

1. Implement the segmentation/MDL search procedure (potentially as a separate
   optimisation stage) using the complexity diagnostics exposed by the warp.
2. Extend adversarial training with the log-sum-exp surrogate and integrate
   validation-based hyper-parameter tuning for the adversarial radius.
3. Provide utilities for estimating spectral templates from disjoint data splits
   to satisfy assumption H-6.

