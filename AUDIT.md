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
| Adversarial robustness (Ω-5′) | **△ Partial** | Training loop supports TR/PGD-style perturbations but does not yet expose the log-sum-exp surrogate objective. |
| Coverage calibration (Ω-2′/Ω-3′) | **✗ Missing** | Block conformal calibration utilities are not yet implemented. |
| Segmentation & MDL penalties (Def. 1, Thm. E′) | **✗ Missing** | Time-warp component exposes anchors and complexity terms but there is no full segmentation/MDL search routine. |
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
   - `Trainer` gains a PGD-style adversarial augmentation step respecting the
     diffeomorphic neighbourhood (ε, ε_log). This partially addresses the
     robustness requirement but the smooth log-sum-exp relaxation from §2.5 is
     still pending.
   - Training metrics now log all regularisers, including calendar and H-disc
     terms.

5. **Deployment Gate**
   - The `DeploymentGate` adopts a statistical predictive interval (SPI) style
     bound based on the empirical variance of loss differences. Additional gates
     (permutation/bootstrap) remain to be implemented.

6. **Missing Components / Future Work**
   - **Coverage calibration:** conformal block calibration operators and the raw
     domain monotonicity mapping described in Ω-2′/Ω-3′ are not yet coded.
   - **Segmentation & MDL:** while the warp exposes anchors and complexity
     measures, the actual MDL-driven segmentation algorithm is absent.
   - **Template estimation:** external estimation of the soft-anchor template is
     assumed available but not implemented in-code.

## Recommendations

1. Add a dedicated conformal calibration module that computes block-wise scores
   in the canonical domain and maps calibrated intervals back via ν⁻¹ and τ.
2. Implement the segmentation/MDL search procedure (potentially as a separate
   optimisation stage) using the complexity diagnostics exposed by the warp.
3. Extend adversarial training with the log-sum-exp surrogate and integrate
   validation-based hyper-parameter tuning for the adversarial radius.
4. Provide utilities for estimating spectral templates from disjoint data splits
   to satisfy assumption H-6.

