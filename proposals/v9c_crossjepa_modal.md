# v9c — Cross-Modal Conformal-JEPA for Brain MRI

**Status**: proposal (post-v9b, post-v8 production).
**Author**: derived from conversation 2026-05-31. Cite CrossJEPA (Nazar et al., arXiv 2511.18424, Nov 2025).
**Relationship to other versions**:
- v8: current production. 2D ConvNeXt-Tiny U-Net, supervised, 384 px, threshold 0.20 + TTA + conformal-CF.
- v9b: normative I-JEPA on healthy 2D slices + latent DDPM + SDF tower + weighted-conformal residual (already coded, awaiting Colab run — see [v9b_IMPLEMENTATION_STATUS.md](v9b_IMPLEMENTATION_STATUS.md) and the runbook section at the bottom of *this* file).
- v9c (this proposal): replaces v9b's self-supervised mask-based JEPA with **cross-modal JEPA** in two directions: (a) 3D volume → 2D slice, and (b) one MRI modality (T1/T1c/T2/FLAIR) → another. Same conformal head; different upstream tower.
- v10 (parked): hyperbolic/causal universal model — separate research line.

## TL;DR

CrossJEPA (Nov 2025) proved three things on point-cloud → image: (1) **masking is not intrinsic to JEPA** — when you have two modalities you already have a context/target split for free; (2) **conditioning the predictor on cross-domain "nuisance" inputs (pose, color) acts as a gradient sink** that lets the encoder learn semantic invariants; (3) **the target-side teacher MUST be frozen** — every learnable-teacher variant they tried collapsed.

v9c applies those three findings to brain MRI in two complementary ways:

- **Method 1 (3D → 2D)**: a learnable 3D ViT encodes the full volume; a predictor reconstructs the *embedding* of any chosen 2D slice as seen by a **frozen 2D teacher = our shipped v8 ConvNeXt-Tiny encoder**. Conditioning = slice plane (axial/sagittal/coronal) + slice index + voxel spacing. Loss = smooth-L1 in v8's embedding space.
- **Method 2 (channel → channel)**: a learnable encoder takes any non-empty subset *S* ⊂ {T1, T1c, T2, FLAIR} of MR sequences as context; a predictor reconstructs the embedding of a held-out modality *m* ∉ *S* as seen by a **frozen m-specific teacher** (one per modality, each pretrained via vanilla I-JEPA on healthy brains in that modality). Conditioning = sinusoidal `target_modality_id` (the gradient sink) + slice-pose + intensity histogram of *S*.

Anomaly score for the v9c head = weighted-conformal-certified residual from one or both towers, reusing v9b's existing conformal infrastructure unchanged.

## Why CrossJEPA validates this design

| CrossJEPA finding | How v9c uses it |
|---|---|
| Masking is not required when you have a cross-modal split | Method 1's split is "3D volume vs. its rendered 2D slices". Method 2's split is "modality subset *S* vs. held-out modality *m*". No masking needed in either; we drop the I-JEPA mask sampler from v9b. |
| Gradient sink: condition predictor on modality-specific nuisances | Method 1 conditions on `(plane, slice_idx, spacing)`. Method 2 conditions on `target_modality_id` + per-context intensity histogram. The encoder is forced to learn anatomy-invariant features because pose/modality identity gets absorbed by the predictor. |
| Frozen teacher is mandatory; learnable target collapses | Method 1: frozen v8 ConvNeXt-Tiny (already shipped). Method 2: 4× frozen modality-specific I-JEPA encoders (one-time pretraining cost). Never train the target side. |
| Cache target embeddings once | Method 1: cache v8 embedding per 2D slice of every BraTS/LGG/Figshare volume. Method 2: cache frozen-teacher embedding per `(modality, patch)`. CrossJEPA reported 16 h/epoch → 2 min/epoch from this trick; the same gain applies here. |

## Method 1: 3D volume → 2D slice (frozen v8 teacher)

### Architecture

```
3D MRI volume (240×240×155, 4-channel)
        │
        ▼
[ 3D ViT-S ] patch=16³, embed=384, depth=12  ← learnable
        │  context tokens
        ▼
[ Predictor ] 6-layer transformer, predictor_dim=192
   ▲                                      │
   │  conditioning:                       │  predicted v8 embedding (B, 1024)
   │  - plane ∈ {axial, sag, cor}         │
   │  - normalised slice index            │
   │  - voxel spacing (1/3 sinusoidal)    │
   │  - intensity histogram (48-d)        │
   │  (gradient sink, see CrossJEPA §D)   │
                                          ▼
                            v8 ConvNeXt-Tiny encoder  ← FROZEN, target
                            (the same one running in production at
                             model/best_micro.onnx)
                                          ▲
                                          │
                            2D rendered slice from the same volume
```

**Loss**: `smooth_l1( pred, frozen_v8_embedding(slice) )` per sampled (plane, slice_idx).

**Per-epoch**: sample 4-8 slices per volume across the three planes, weighted toward the brain's central z-range (where there is more structure).

### Training pool (no labels required)

- **BraTS 2020** (369 patients): all 4 modalities co-registered.
- **LGG (kaggle_3m, 110 patients)**: T1/T1ce/FLAIR.
- **Figshare-Cheng (3064 slices, ~233 patients via slice-stem grouping)**: T1c mostly.

**Volumes we can recover** from existing dataset_v8 prep ≈ 700+. Kaggle 4-class slices are loose (no volume parent) → excluded from Method 1 (still usable for v9b's healthy-only pretrain).

### Why this is good for us specifically

- **Zero new teacher to find or train**. v8 is already a strong, in-distribution 2D MRI encoder; we just freeze it and ask "what would v8 see if it looked at axial slice #87 of this volume?".
- **The 3D encoder learns a clinically-useful representation by construction**: it has to predict what each 2D plane will look like, which is the same skill a radiologist uses when scrolling through a stack.
- **At inference**, the 3D encoder produces a per-voxel anomaly map by comparing its predicted-slice-embedding to v8's actual embedding for that slice; deviations from the learned "healthy-volume → expected-2D-view" function flag anomalies.

### Why not 2D → 3D direction?

It's tempting to add the reverse direction for symmetry (one 2D slice → 3D embedding), but CrossJEPA explicitly warns that dual-direction setups collapse (P2I+I2I → 91.7 % with collapse; P2I+P2P → 92.0 % with gradient imbalance vs. P2I-only at 94.2 %). The 2D → 3D direction is also fundamentally harder (slice-to-volume is generative, not predictive — a single 2D slice doesn't contain the cross-slice information). **Stick to single-direction with frozen teacher.**

## Method 2: cross-modality JEPA (T1/T1c/T2/FLAIR)

### Architecture

```
Context: any non-empty subset S ⊂ {T1, T1c, T2, FLAIR}
        │
        ▼
[ Per-modality patch embedder ]  (4 small adapters, one per channel)
        │  concatenated context tokens
        ▼
[ Shared ViT-S backbone ] patch=16, embed=384, depth=12  ← learnable
        │
        ▼
[ Predictor ] 6-layer, predictor_dim=192
   ▲                                      │
   │  conditioning:                       │  predicted target embedding (B, 192)
   │  - target_modality_id sinusoidal     │
   │    (the gradient sink)               │
   │  - slice-pose / patch position       │
   │  - intensity histogram of S (per     │
   │    channel, concatenated)            │
                                          ▼
                            Frozen target teacher  ← one per modality
                            (4 separate ViT-S, each pretrained via
                             vanilla I-JEPA on healthy brains in that
                             modality only — one-time cost)
                                          ▲
                                          │
                            Held-out modality m ∉ S
```

### Training data

BraTS gives 4-modality co-registered volumes for ~1500 patients (BraTS 2020 + 2021 if added). Per scan, the valid (context-subset, target) pairs are:

- 2⁴ − 1 = 15 non-empty subsets *S*
- 4 possible targets *m*
- valid only when *m* ∉ *S* → 15 × 4 − 15 = 45 valid (S, m) pairs per scan
- in practice we sample uniformly: pick *m* first, then a random subset of {T1,T1c,T2,FLAIR}\{m}
- → ~22 k training tuples per epoch with random subset sampling on BraTS alone

### The novelty bit

Nobody has yet combined all four of:
- BraTS-style channel modality split (T1/T1c/T2/FLAIR)
- JEPA-style **latent** prediction (not pixel reconstruction)
- **Frozen per-modality teacher** (CrossJEPA's anti-collapse design)
- Gradient-sink conditioning on `target_modality_id` + slice-pose

Existing related work and how v9c differs:

| Paper | What they do | Why v9c differs |
|---|---|---|
| MultiMAE-for-MRI (2509.11442) | Treat T1/T1c/T2/FLAIR as separate modalities, mask at patch + full-modality level (Dirichlet α=1), reconstruct **pixels** with MSE | v9c reconstructs **embeddings**, not pixels; uses frozen teacher; uses gradient-sink conditioning; no masking. |
| M³FeCon "Missing as Masking" (MICCAI 2024) | Cross-modal **feature** reconstruction across arbitrary subsets — closest in spirit | Both encoder + decoder are **learnable** in M³FeCon → vulnerable to CrossJEPA's documented collapse modes. v9c freezes the target side. |
| CrossJEPA (2511.18424) | 3D point cloud → 2D image | v9c applies the same recipe to MRI channels — the original paper didn't touch a multi-channel modality structure. |

### Honest practical caveats

1. **Only BraTS gives the 4-modality paired ground truth** for Method 2. Kaggle has unlabeled 2D PNGs with unknown modality; Figshare is mostly T1c. So Method 2's pretraining pool is ~1500 patients, not 6000+ slices. Still enough.
2. **Method 1 doesn't need modality labels** — works on the full dataset_v8 pool as long as you can recover the volume each slice came from. Pool ≈ BraTS + LGG + Figshare ≈ 2k volumes.
3. **Compute**: 3D ViT over 240³ at patch=16 = 15³ ≈ 3,375 tokens per pass, depth=12, embed=384 — manageable on A100, tight on T4. Plan **Colab Pro+ A100 / H100**.
4. **Caching is huge**. Cache (a) v8 ConvNeXt embedding per slice for Method 1 and (b) frozen-teacher embedding per (modality, patch) for Method 2. Both computable once at the start of training and reused every epoch — same trick that took CrossJEPA from 16 h/epoch to 2 min/epoch.
5. **Stick to single-direction with frozen teacher.** No "predict 2D from 3D AND 3D from 2D simultaneously" — CrossJEPA's dual-branch experiments all collapsed.

## How v9c plugs into the existing v9b stack

v9b already provides:
- weighted conformal calibrator ([src/research/jepa_conformal.py](../src/research/jepa_conformal.py))
- mesh extraction ([src/research/mesh_extraction.py](../src/research/mesh_extraction.py))
- MNI152 atlas registration ([src/research/mni152_registration.py](../src/research/mni152_registration.py))
- two-tower combiner ([src/research/two_tower_anomaly.py](../src/research/two_tower_anomaly.py))

v9c **swaps the v9b JEPA tower for one or both CrossJEPA towers**; the conformal certification, mesh, atlas, and combiner layers are reused unchanged. That keeps the implementation cost focused on the new pretraining loops.

## File layout (when we build it)

```
src/research/v9c_crossjepa/
  __init__.py
  volume_to_slice.py        # Method 1: 3D ViT + predictor + v8 frozen target
  modality_to_modality.py   # Method 2: subset encoder + predictor + frozen-m target
  conditioning.py           # sinusoidal pose/plane/modality-id encoders
  caching.py                # one-time v8 + frozen-teacher embedding cache
src/train_v9c_method1_vol2slice.py
src/train_v9c_method2_mod2mod.py
src/v9c_inference.py        # weighted-conformal score from both towers
tests/test_v9c_components.py
colab_bundle/v9c_colab_train.ipynb
```

## Sources

- CrossJEPA paper: <https://arxiv.org/abs/2511.18424>
- CrossJEPA HTML full text: <https://arxiv.org/html/2511.18424v1>
- MultiMAE for Brain MRIs: <https://arxiv.org/html/2509.11442v1>
- Missing as Masking / M³FeCon (MICCAI 2024): <https://papers.miccai.org/miccai-2024/520-Paper0067.html>
- Multimodal MAE 3D MRI tumor analysis: <https://arxiv.org/html/2505.00568v1>
- 3D-JEPA self-supervised 3D representation: <https://arxiv.org/html/2409.15803v1>
