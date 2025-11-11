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
| Adversarial robustness (Ω-5′) | **✓ Implemented** | PGD-based diffeomorphic perturbations together with the log-sum-exp surrogate penalty for \(\tilde{\mathcal R}_{\rm adv}\). |
| Coverage calibration (Ω-2′/Ω-3′) | **✓ Implemented** | Canonical-domain block conformal calibration mapping intervals back through \(\nu^{-1}\) and \(\tau\). |
| Segmentation & MDL penalties (Def. 1, Thm. E′) | **✓ Implemented** | MDL-style penalties derived from warp anchors and value transform bins integrated into the training objective. |
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
   - `Trainer` now augments the PGD routine with a differentiable log-sum-exp
     surrogate that upper-bounds the worst-case risk while retaining gradients
     for model parameters.
   - Adversarial penalties and MDL terms are reported alongside other
     regularisers, keeping optimisation diagnostics comprehensive.

5. **Coverage Calibration**
   - `BlockConformalCalibrator` implements blockwise canonical residual scoring,
     optional β-mixing corrections and faithful mapping of calibrated intervals
     back to the observation domain.

6. **Segmentation & MDL Penalties**
   - `mdl_penalty` synthesises the structural MDL terms from warp anchors and
     value-transform knot counts and is consumed inside the training loop.

7. **Deployment Gate**
   - The `DeploymentGate` retains the SPI-style bound for guarding deployment;
     permutation/bootstrap variants remain future work.

8. **Remaining Gaps / Future Work**
   - **Template estimation:** external estimation of the soft-anchor template is
     assumed available but not implemented in-code.
   - **Extended gating:** adding permutation or bootstrap-based risk difference
     bounds would complete the Ω-4′ toolbox.

## Recommendations

1. Add a dedicated conformal calibration module that computes block-wise scores
   in the canonical domain and maps calibrated intervals back via ν⁻¹ and τ.
2. Implement the segmentation/MDL search procedure (potentially as a separate
   optimisation stage) using the complexity diagnostics exposed by the warp.
3. Extend adversarial training with the log-sum-exp surrogate and integrate
   validation-based hyper-parameter tuning for the adversarial radius.
4. Provide utilities for estimating spectral templates from disjoint data splits
   to satisfy assumption H-6.

