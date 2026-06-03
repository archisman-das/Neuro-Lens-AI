# v9c CrossJEPA — Implementation Status

Tracking the build-out of [v9c_crossjepa_modal.md](v9c_crossjepa_modal.md).

> **Naming note** — the previously-shipped v9c (DINOv2 + JEPA predictor,
> `src/research/v9c_dinov2_jepa.py`) is a *different* model. To avoid
> confusion the CrossJEPA implementation lives in its own package:
> `src/research/v9c_crossjepa/`. Both can coexist.

## Status (this PR)

| Component | Status | Notes |
|---|---|---|
| `src/research/v9c_crossjepa/__init__.py` | ✅ shipped | Package init + module list |
| `src/research/v9c_crossjepa/conditioning.py` | ✅ shipped | Plane/modality/spacing/histogram embeddings; Method 1 + Method 2 conditioning modules (the gradient sink) |
| `src/research/v9c_crossjepa/caching.py` | ✅ shipped | Disk-backed teacher-embedding cache (atomic-write `.npz`, SHA-1 keyed, 256-way subdirs for Drive friendliness) |
| `src/research/v9c_crossjepa/v8_teacher.py` | ✅ shipped | Frozen v8 ConvNeXt-Tiny encoder wrapper. Loads from the shipped UNet checkpoint, peels off the encoder, asserts `requires_grad=False` on every param |
| `src/research/v9c_crossjepa/vit_3d.py` | ✅ shipped | 3D patch embed (Conv3d) + 3D sin-cos posemb + ViT3DEncoder trunk (reuses v9b's `TransformerBlock`) |
| `src/research/v9c_crossjepa/volume_to_slice.py` | ✅ shipped | Method 1 model: 3D ViT → predictor (cross-attention over volume tokens with conditioning gradient-sink) → predicts frozen v8 embedding of any target slice. smooth-L1 loss |
| `src/research/v9c_crossjepa/modality_to_modality.py` | ✅ shipped | Method 2 model: per-modality patch embedders + shared backbone + cross-attention predictor + 4-frozen-teacher invariant |
| `src/research/v9c_crossjepa/dataset_3d.py` | ✅ shipped | Vol2SliceDataset + Mod2ModDataset (iterable, NIfTI streaming, robust intensity normalization, `discover_brats_scans` helper, `collate_mod2mod` groups by target modality) |
| `src/train_v9c_method1_vol2slice.py` | ✅ shipped | Method 1 training loop: load v8 teacher, build 3D ViT, smooth-L1 vs cached/live v8 embedding, atomic checkpoints, AMP support |
| `src/train_v9c_method2_mod2mod.py` | ✅ shipped | Method 2 training loop: loads 4 per-modality I-JEPA teachers (raises clear error if any missing), runs cross-modality JEPA |
| `tests/test_v9c_crossjepa_components.py` | ✅ shipped | 13 tests, all passing — shapes, frozen-teacher invariants, gradient sink, "teacher weights don't change after backward" |
| Pretrain script for 4 per-modality I-JEPA teachers | ⏳ deferred | Reuse `src/train_v9b_stage1_jepa.py` with `--modality_filter T1` (etc.) for each. One-time job per modality on Colab A100. ~6-8 h each. |
| `src/v9c_crossjepa_inference.py` | ⏳ deferred to next PR | Two-tower combiner integration once a trained checkpoint exists |
| `colab_bundle/v9c_crossjepa_train.ipynb` | ✅ shipped | End-to-end Colab notebook: pull BraTS via HF datasets, build v8 teacher, train Method 1 |
| Trained checkpoints | ❌ pending | Awaits Colab Pro+ A100/H100 run (Method 1 ~10-15h, Method 2 ~15-20h after 4 teacher pretrains) |

## What this PR does NOT do (and why)

1. **No trained weights.** Pretraining is a multi-hour Colab job; this PR ships the trainable code only.
2. **No inference integration into `dashboard.py` / `v9b_advisory.py`.** That requires real checkpoints + a weighted-conformal calibration pass; deferred to a follow-up PR.
3. **No 4 per-modality I-JEPA teachers.** Each is a separate ~6-8h training. The Method 2 trainer raises a clear error if a teacher .pt is missing, so the failure mode is obvious.

## How to use it

### Method 1 (3D → 2D), single Colab session

```python
# In a Colab cell (after running the bundle's setup cell):
!python src/train_v9c_method1_vol2slice.py \
    --scans_glob '/content/brats/*.nii.gz' \
    --v8_ckpt /content/v8_unet.pt \
    --output_dir /content/drive/MyDrive/v9c_crossjepa_method1 \
    --volume_size 144 192 192 \
    --in_channels 4 \
    --batch_size 2 --epochs 30 --lr 2e-4 \
    --num_workers 2 --amp --resume auto
```

### Method 2 (modality → modality), requires teachers first

```python
# Phase A — one-time, per modality (reuses existing v9b I-JEPA trainer):
for m in ['T1', 'T1c', 'T2', 'FLAIR']:
    !python src/train_v9b_stage1_jepa.py \
        --data_dir /content/brats_${m}_healthy_slices \
        --output_dir /content/teachers/${m} \
        --epochs 50 --batch_size 16 --amp --resume auto

# Phase B — CrossJEPA Method 2:
!python src/train_v9c_method2_mod2mod.py \
    --brats_root /content/brats \
    --teachers_dir /content/teachers \
    --output_dir /content/drive/MyDrive/v9c_crossjepa_method2 \
    --image_size 256 --batch_size 8 --epochs 30 --lr 2e-4 --amp
```

## CrossJEPA invariants enforced in code

1. **Frozen teacher** — `V8FrozenTeacher.__init__` runs `requires_grad_(False)` + `_assert_frozen()` on construction. `Vol2SliceModel` holds the teacher as a plain attribute (NOT a submodule) so `.parameters()` never returns teacher params. Same pattern in `Mod2ModModel`. The test `test_vol2slice_teacher_frozen` and `test_mod2mod_teachers_dont_get_gradient` actively verify this.

2. **No masking** — neither Method 1 nor Method 2 has a mask sampler. The context/target split IS the cross-modal split (volume↔slice for M1, subset↔held-out for M2).

3. **Single direction** — neither model has a reverse path. CrossJEPA's paper documented every dual-direction variant collapsing (P2I+I2I at 91.7%, P2I+P2P at 92.0%, vs P2I-only at 94.2%).

4. **Gradient sink via conditioning** — the predictor's query token starts from the conditioning vector (plane/slice/spacing/hist for M1, modality_id/pose/hist for M2). The predictor absorbs these nuisances; the encoder is forced to learn anatomy-invariant features.

5. **Cache target embeddings once** — `caching.TeacherEmbeddingCache` is content-addressed and idempotent. CrossJEPA reports 16h → 2min/epoch from this; the v8 teacher embed is ~30 ms/slice on GPU, so caching the full BraTS pool (369 patients × 3 planes × 6 picks ≈ 6600 slices) takes ~3 minutes once and saves all subsequent epochs.
