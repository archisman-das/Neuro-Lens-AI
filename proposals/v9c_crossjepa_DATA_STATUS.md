# v9c CrossJEPA — Data Status

Tracking what 3D MRI data is available locally for training, and what
needs Colab/Drive access.

## Local pools (ready to train against)

| Cohort | Patients | Disk | Path | Use |
|---|---:|---:|---|---|
| **radiata-ai/brain-structure** | 800 healthy | 1.7 GB on `C:` (HF cache) | `~/.cache/huggingface/hub/datasets--radiata-ai--brain-structure/snapshots/aea73e2.../{DLBS,NKI-RS,OASIS-1,OASIS-2}/sub-*/ses-*/anat/*.nii.gz` | Method 1 baseline pretrain (any 3D volume) + clean negatives for conformal calibration |
| **obi77/brats23-first-10-examples** | 10 tumor | 130 MB on `D:` | `D:/datasets/brats23-10-examples/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` | Method 2 smoke-test (BraTS-2023 dash naming, 10/10 fully-modal) |
| **rocky93/BraTS_segmentation** | 1251 tumor (1248 fully-modal) | 12 GB on `D:` | `D:/datasets/brats2021/BraTS2021_*/` | Method 2 full training (T1/T1c/T2/FLAIR + seg) |

Total tumor 3D volumes locally: **1261** (1248 fully-modal). Total healthy: **800**.

## Per-modality coverage in `D:/datasets/brats2021/`

| Modality | Patients |
|---|---:|
| T1 | 1249 |
| T1c | 1251 |
| T2 | 1251 |
| FLAIR | 1250 |
| Fully 4-modal (all of the above) | 1248 |

## Verified end-to-end

- `Vol2SliceDataset` streams real radiata-ai NIfTI cleanly at ~80 ms/batch on local CPU. Voxel spacing detected (1.5 mm iso, MNI). All 3 planes (axial/sagittal/coronal) sampled, intensity histograms sum to 1.0 per slice. `slice_resize_hw=(256, 256)` so axial/sagittal/coronal collate without shape conflicts.
- `Mod2ModDataset` + `collate_mod2mod` streams real BraTS-2021 cleanly. Context subsets vary per sample, target modality rotates across the 4. ~1.7 s/batch on CPU at 128 px (will be much faster on GPU + with more workers).
- `discover_brats_scans` handles **both** BraTS-2020/2021 (`_t1ce.nii.gz` underscore convention) and **BraTS-2023** (`-t1c.nii.gz` dash convention) in the same call.

## What still needs Colab / Drive

| Item | Why | Mitigation |
|---|---|---|
| Method 1 actual training | 3D ViT on (144, 192, 192, 4) at depth=12 = ~30 GB GPU memory at batch=2 | Use Colab A100 (40 GB) or H100. The trainer is fully Colab-ready (`scripts/train_v9c_method1_vol2slice.py --amp`). |
| Per-modality I-JEPA teachers (4 of them, one per modality) | Method 2 requires frozen teachers; each is a separate ~6-8h pretrain | Reuse `src/train_v9b_stage1_jepa.py` with a per-modality filter, one Colab session per modality |
| Method 2 actual training | Needs all 4 teachers first | Sequential after teachers are pretrained |
| Frozen v8 ConvNeXt-Tiny `.pt` for Method 1 teacher | Currently only `model/best_micro.onnx` + `.pt` exists locally; the teacher loader expects a PyTorch state dict in segmentation-models-pytorch UNet format | The shipped `model/best_micro.pt` is the right file — `V8FrozenTeacher.from_unet_checkpoint('model/best_micro.pt')` works |

## Disk pressure snapshot

```
D: 250G  195G used  56G free  (after BraTS-2021 pull)
E: 301G  257G used  45G free
C: 400G  399G used   920M free   <-- KEEP CLEAR; do not download more to C:
```
