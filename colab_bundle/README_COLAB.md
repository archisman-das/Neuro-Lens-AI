# v8 Brain Tumor Training on Google Colab

## What you upload, where

**To Google Drive** (one-time, persistent):
| File | Location on Drive | Size | Why |
|---|---|---|---|
| `dataset_v8.zip` | `MyDrive/neurolens/dataset_v8.zip` | ~860 MB | The training data (BraTS + LGG + Figshare + Kaggle no-tumor) |
| `colab_bundle.zip` | `MyDrive/neurolens/colab_bundle.zip` | ~30 KB | The trainer code (src/train_segmentation_v7.py + dependencies) |

**To Colab runtime** (each session):
| File | How | Why |
|---|---|---|
| `v8_colab_train.ipynb` | Open via File → Upload notebook | The notebook itself |

The notebook auto-creates `MyDrive/neurolens/attention_unet_v8/` on Drive — this is where checkpoints land. Survives Colab disconnects.

## Step-by-step

### 1. Create the two zips on your laptop

```bash
# Inside e:/Neuro-Lens-AI-main/Neuro-Lens-AI-main/

# Zip the dataset (~860 MB)
cd dataset_v8/.. && zip -r dataset_v8.zip dataset_v8/

# Zip the code bundle
cd colab_bundle && zip -r ../colab_bundle.zip train_segmentation_v7.py train_segmentation_v5.py __init__.py requirements_colab.txt && cd ..
```

(On Windows PowerShell: use `Compress-Archive -Path dataset_v8 -DestinationPath dataset_v8.zip` instead of `zip -r`.)

### 2. Upload to Drive

- Open drive.google.com
- Create folder `neurolens` in `My Drive`
- Upload `dataset_v8.zip` and `colab_bundle.zip` into it
- Wait for both uploads to complete (dataset_v8.zip is slow due to size; ~15-30 min on typical home internet)

### 3. Open Colab

- Go to colab.research.google.com
- File → Upload notebook → select `v8_colab_train.ipynb`
- Runtime → Change runtime type → **GPU**:
  - Free: T4 (15 GB VRAM, ~24 hr training)
  - Pro: L4 or A100 (much faster)

### 4. Run all cells

The notebook handles everything:
- Mounts Drive
- Extracts dataset_v8 to local Colab disk (faster I/O than reading Drive directly)
- Extracts code bundle
- Installs deps (segmentation-models-pytorch + timm)
- Trains with --resume auto pointing at Drive checkpoint dir
- Atomic checkpoint writes survive disconnects

### 5. When Colab disconnects (free tier: every ~12 hr)

Just **rerun cell 7** (the training cell). The `--resume auto` flag picks up `last.pt` from Drive and continues from the exact step where it stopped (max ~2-5 min of work lost per disconnect due to `--checkpoint_every_steps 500`).

Expected sessions to complete on free T4:
- Session 1: ~12 hr → epoch ~30/60
- Session 2: ~12 hr → epoch 60/60 done
- Total: 2 sessions, all auto-resumed

## What's in colab_bundle.zip

| File | Purpose |
|---|---|
| `train_segmentation_v7.py` | The crash-safe trainer (atomic checkpoints, --resume auto, intra-epoch saves) |
| `train_segmentation_v5.py` | Parent class for V5SegDataset (the v7 dataset extends it) |
| `__init__.py` | Makes the src/ directory a Python package |
| `requirements_colab.txt` | Pip deps to install in Colab |
| `v8_colab_train.ipynb` | The notebook itself (also place in Drive for backup) |
| `README_COLAB.md` | This file |

## Expected runtime

| GPU | Per epoch | Total (60 epochs) | Sessions needed (free Colab disconnects ~12 hr) |
|---|---|---|---|
| T4 (free) | ~25 min | ~24 hr | 2 sessions, auto-resume |
| L4 (Pro) | ~12 min | ~12 hr | 1 session |
| A100 (Pro+) | ~6 min | ~6 hr | 1 session |
| H100 (paid pods) | ~3 min | ~3 hr | 1 session |

## After training

- `best_micro.pt` → highest micro-Dice checkpoint (the headline number you want)
- `best_model.pt` → highest composite (dice − 5·fp_rate) checkpoint (production-safest)
- `last.pt` → most recent checkpoint
- `training.log` → per-epoch metrics for plots
- All in `MyDrive/neurolens/attention_unet_v8/` on Drive

Cell 9 in the notebook exports `best_micro.pt` to ONNX for deployment. Cell 10 downloads everything as a zip.

## Cost

Free tier: **$0** (T4 GPU, 12-hr sessions, generally enough for v8).

If you want faster:
- Colab Pro: $10/mo, L4/A100 access, 24-hr sessions
- Colab Pro+: $50/mo, more compute units, A100 priority

For v8 training specifically, **free tier is sufficient** (2 sessions, no cost).
