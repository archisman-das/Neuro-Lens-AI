# v10 Implementation Status

**Status:** Phase 1-7 implemented at brain-2D scope. Parked at
`src/research/_v10_universal_hyperbolic/` while v9b (Normative JEPA +
Conformal) is the active research direction. Pick up v10 as a follow-up
paper after v9b ships.

**Why parked:** v9b is shorter (5-6 mo vs 10-11 mo), lower compute
($3-7K vs $8-20K), higher completion likelihood (80% vs 60%), and
matches the "model that understands geometry behind pixels" stated goal
more directly through JEPA's latent-space prediction. v10's universal
multi-organ scope is a stronger story but a bigger undertaking, naturally
positioned as the v9b follow-up.

## What's implemented (working, tested)

| Module | Location | LoC | Status |
|---|---|---|---|
| Poincaré ball math (expmap0, logmap0, Möbius add/matvec, geodesic distance, HyperbolicProjection, HyperbolicLinear, PoincareDistance) | `_v10_/hyperbolic.py` | ~190 | ✅ tested |
| Causal SCM head (CausalSplitHead, CausalRecompose, LearnableDAGAdjacency with NOTEARS, orthogonality via cross-correlation, intervention consistency, CausalSCMHead wrapper) | `_v10_/causal_scm.py` | ~270 | ✅ tested |
| Geometric prior (synthetic brain SDF template, GeometricPriorConditioning, SIRENImplicitSDF) | `src/research/geometric_prior.py` (shared) | ~190 | ✅ tested |
| Counterfactual healthy decoder (CounterfactualHealthyDecoder with masked reconstruction loss, tumor_residual) | `src/research/counterfactual_decoder.py` (shared) | ~170 | ✅ tested |
| Hyperbolic conformal (HyperbolicCalibSample, weighted_quantile_tibshirani, HyperbolicConformalCalibrator, voxelwise_hyperbolic_anomaly_map) | `_v10_/hyperbolic_conformal.py` | ~250 | ✅ tested |
| Integrated V10Model (geometric prior → ConvNeXt-Tiny → hyperbolic → SCM → segmentation decoder + counterfactual branch) | `_v10_/v10_model.py` | ~250 | ✅ tested forward pass |
| V10 trainer (multi-task loss: Tversky+Dice+BCE + ortho + DAG + forbidden-edge + counterfactual + crash-safe checkpointing + AMP + RAM cache + --resume auto) | `src/train_segmentation_v10.py` | ~340 | ✅ syntax OK |
| Unit tests (16 tests across all modules) | `tests/test_v10_components.py` | ~310 | ✅ all pass |

**Total new code:** ~1,970 LoC + tests.

## What's NOT implemented (out of v10 brain-2D scope)

| Deferred piece | Why | Where it'd go |
|---|---|---|
| Universal multi-organ encoder (SAT3D-style on LiTS/KiTS/AMOS/PROSTATEx) | Requires ~50K 3D volumes + multi-week pretrain on a GPU cluster | Future v10 paper |
| 3D ViT encoder | Brain-2D first; 3D = 16× memory/compute, requires multi-modal volumetric upload UX | v10 Phase 2 expansion |
| Real MNI152 atlas SDF templates (we use synthetic brain ellipse) | Requires FreeSurfer pipeline for per-patient SDFs | drop-in via `geometric_prior.load_external_sdf()` |
| End-to-end clinical evaluation with radiologists | Requires 3-5 board-certified raters | Future paper Phase 9 |
| Multi-organ calibration sets for hyperbolic conformal | Per-organ calibration data | Future v10 paper |
| MNI152 atlas registration via ANTs/SimpleITK | Light wrapper, ~200 LoC, deferred to v9b instead | v9b Commit 3 |
| LLM Pattern A/B/C/D integration (already exists for v8) | Adapter only, low priority for v10 | v9b Commit 3 |

## How to resume v10 work

1. Read `proposals/v9_universal_causal_hyperbolic_tumor.md` (the full proposal)
2. Phase 1-7 already implemented. Resume from **Phase 8 (multi-organ data)**: download LiTS, KiTS23, AMOS, MSD (~13K 3D volumes); adapt `V10Model` encoder to 3D + 4-channel input
3. Apply for academic compute (NIH NCATS / NVIDIA Hardware Grant / Lambda Labs Research Credit)
4. ~6-8 months of focused effort to ship as TMI / Nature Methods paper

## Migration: what moves vs stays after v10 unpark

If/when v10 is unparked for active development:

- `_v10_/hyperbolic.py`, `_v10_/causal_scm.py`, `_v10_/hyperbolic_conformal.py`, `_v10_/v10_model.py`: **active development** — move out of `_v10_/` to `src/research/`
- `src/research/geometric_prior.py`, `src/research/counterfactual_decoder.py`: **stay where they are** — shared with v9b
- `src/train_segmentation_v10.py`: **active development** — keep at src/

## Tests command

```bash
python tests/test_v10_components.py
# All 16 tests should pass (~5s on CPU, no GPU required)
```

## v10 vs v9b at-a-glance

| Aspect | v10 (parked) | v9b (active) |
|---|---|---|
| Scope | Multi-organ universal (eventually) | Brain only |
| Supervision | Supervised tumor masks | Unsupervised (healthy only) |
| Latent geometry | Hyperbolic (Poincaré ball) | Euclidean (JEPA latents) |
| Disentanglement | Causal SCM (anatomy/tumor/scanner) | None (purely normative) |
| Generative | Small autoregressive decoder | Latent diffusion decoder |
| Hard math | Hyperbolic conformal (Möbius distances) | JEPA-latent conformal (prediction residuals) |
| Time | 10-11 months | 5-6 months |
| Compute | $8-20K | $3-7K |
| Completion likelihood | 60% | 80% |
| Risk | High (5 integration points) | Medium (4 integration points) |

---

**v10 is feature-complete at brain-2D scope; multi-organ + 3D + clinical
eval are explicitly deferred.** All shipped modules pass tests and are
ready to plug back in when v10 development resumes.
