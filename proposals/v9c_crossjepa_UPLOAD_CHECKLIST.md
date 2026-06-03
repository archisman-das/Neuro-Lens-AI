# v9c CrossJEPA — Drive upload checklist

Single source of truth for which files you need on Drive to run each
training notebook on Colab. Build them once locally; upload them once;
re-use across sessions.

## Method 1 v2 (with preprocessing-invariance fix)

| Drive path | Local source | Built by | Size |
|---|---|---|---:|
| `MyDrive/colab_bundle.zip` | `E:/Neuro-Lens-AI-main/Neuro-Lens-AI-main/colab_bundle.zip` | `scripts/rebundle_colab.py` | 102 KB |
| `MyDrive/crossjepa_data_v2.zip` | `D:/dataset_bundle/crossjepa_data_v2.zip` | `scripts/build_crossjepa_data_v2.py` | 15.57 GB |

The v8 frozen teacher (`attention_unet_v8/best_micro.pt`, 383 MB) is
pulled from HF Models by the notebook automatically — no upload needed.

Notebook: `colab_bundle/v9c_crossjepa_train.ipynb` (also bundled into
`colab_bundle.zip`).

Expected wall-clock: ~10–12h on A100 for 30 epochs.

## Method 2 (full, with 4 per-modality teachers + main run)

| Drive path | Local source | Built by | Size |
|---|---|---|---:|
| `MyDrive/colab_bundle.zip` | same as Method 1 | same | 102 KB |
| `MyDrive/crossjepa_data_v2.zip` | same as Method 1 | same | 15.57 GB |
| `MyDrive/dataset_v9c_modality_teachers.zip` | `D:/dataset_bundle/dataset_v9c_modality_teachers.zip` | `scripts/build_brats_healthy_slices.py` + zip | 1.06 GB |

Notebook: `colab_bundle/v9c_crossjepa_method2_train.ipynb` (also bundled
into `colab_bundle.zip`).

Expected wall-clock:
- Phase A (4 teachers, sequential): ~24–32h on A100 (resume-safe across sessions)
- Phase B (Method 2 main): ~10–15h on A100
- **Total**: ~35–47h across multiple sessions

## Build commands (run once locally before first upload)

```bash
# 1. Source bundle (after any code change)
python scripts/rebundle_colab.py
#    -> ./colab_bundle.zip  (~100 KB, takes <1s)

# 2. Method 1 v2 data bundle (after pulling held-out IXI)
python scripts/build_crossjepa_data_v2.py
#    -> D:/dataset_bundle/crossjepa_data_v2.zip  (~16 GB, takes ~1 min)

# 3. Method 2 per-modality teacher slices (one-time extraction + zip)
python scripts/build_brats_healthy_slices.py
#    -> D:/datasets/dataset_v9c_modality_teachers/{T1,T1c,T2,FLAIR}/*.png
python e:/tmp/zip_teacher_data.py
#    -> D:/dataset_bundle/dataset_v9c_modality_teachers.zip  (~1.1 GB)
```

## What happens if you skip a file

| Skipped upload | Failure mode |
|---|---|
| `colab_bundle.zip` v2 | Notebook can't unzip source → can't import the augmentation code → Method 1 v2 cell will run v1 trainer (no augmentation, repeating the preprocessing-bias failure) |
| `crossjepa_data_v2.zip` | Notebook's cell 4 fails to unzip → no `/content/data/healthy/` → trainer fails with "no scans matched" |
| `dataset_v9c_modality_teachers.zip` (Method 2 only) | Phase A cell 4 fails (no `/content/teacher_data/T1/`) → all 4 teachers fail to start → Phase B raises "Missing teachers" |
