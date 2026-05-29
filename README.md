# NeuroLens AI

NeuroLens AI is a brain MRI analysis project for tumor detection, model comparison, and segmentation. It includes a browser dashboard, a Streamlit interface, TensorFlow training scripts for the classifiers, a PyTorch Attention U-Net for segmentation, evaluation utilities, and research workflows for U-Net experiments.

> This project is for research and educational use only. It is not a medical device and should not be used as the sole basis for clinical decisions.

## Features

- MRI upload + inference for tumor / no-tumor binary classification (CNN, ResNet50 transfer, hybrid ResNet+ViT).
- Real Grad-CAM overlays for CNN and Transfer Learning models in the HTTP dashboard.
- ViT patch-saliency for the hybrid ViT (computed on the patch-token sequence).
- PyTorch **Attention U-Net** for binary tumor segmentation, trained on GPU.
- Browser dashboard (`web_dashboard/`) that talks to the real `/predict` and `/segment` endpoints.
- Streamlit dashboard (`app.py`) for quick local model comparison.
- Reference TF segmentation framework (U-Net / Attention U-Net / Res U-Net / Multi-modal U-Net) with k-fold, ablation, and robustness scripts.
- Documentation under `Documentation/`.

## What this is NOT

The earlier IEEE-style write-up in `Documentation/` describes pure 3D MRI, federated learning, and self-supervised pre-training. **Those features are scaffolding under `src/advanced_models.py` and are not wired into any production code path.** See `PROJECT_DOCUMENTATION.md` for an honest feature table.

## Project Structure

```text
.
|-- app.py                    # Streamlit dashboard
|-- dashboard.py              # Local HTTP dashboard server
|-- web_dashboard/            # HTML, CSS, and JS dashboard UI
|-- src/                      # Models, training, evaluation, explainability, segmentation
|-- config.yaml               # Segmentation and experiment configuration reference
|-- Dashboard_Images/         # Dashboard screenshots/images
|-- Documentation/            # Report files
|-- ppt/                      # Presentation deck
`-- requirements.txt          # Python dependencies
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the HTML Dashboard

```bash
python dashboard.py --port 8501
```

Then open:

```text
http://localhost:8501/
```

The dashboard looks for trained weights and metrics in:

- `real_eval_fixed/`
- `real_eval_current/`
- `artifacts/`

Expected classification weights path:

```text
artifacts/<model_name>/best_weights.weights.h5
```

where `<model_name>` is one of `cnn`, `transfer`, or `vit`.

## Run the Streamlit App

```bash
streamlit run app.py
```

## Train Classification Models

```bash
python src/train.py --model cnn --dataset dataset --epochs 10 --batch_size 32 --output artifacts
```

Available model choices:

- `cnn`
- `transfer`
- `vit`

Training saves weights and history under `artifacts/<model_name>/`.

## Train Segmentation Models (PyTorch, GPU)

The active segmentation pipeline is in PyTorch because TF 2.21 has no native-Windows GPU support. Step 1 generates pseudo-masks from the existing classification dataset (no ground-truth masks ship with the Kaggle source):

```bash
python generate_pseudo_masks.py --source dataset_real --output dataset_real --clean
```

Step 2 trains the Attention U-Net on GPU (CUDA auto-detected; falls back to CPU otherwise):

```bash
python src/train_segmentation_torch.py --data_dir dataset_real \
    --epochs 25 --batch_size 8 --image_size 256 --base_filters 32
```

Outputs land in `segmentation_artifacts/attention_unet/`:

- `best_model.pt` (state dict + config)
- `history.json`, `training_curves.png`
- `evaluation_metrics.json`

The dashboard's `/segment` endpoint loads these weights automatically.

### Reference TensorFlow segmentation (CPU)

The original TF U-Net stack still works for CPU experimentation / k-fold / ablation:

```bash
python src/train_segmentation.py --data_dir dataset_real --model_type attention_unet \
    --epochs 25 --batch_size 8
```

The TF script expects `<split>/images/` and `<split>/masks/` — generate them with `generate_pseudo_masks.py` first.

## Dataset Notes

The Kaggle Brain Tumor MRI dataset is a 2D JPG classification dataset (`tumor` / `no_tumor`). It contains no ground-truth segmentation masks. `generate_pseudo_masks.py` synthesises binary masks via brain-region + Otsu thresholding + largest-blob filtering. These are weakly-supervised pseudo-labels suitable for demoing the U-Net pipeline — they are NOT radiologist annotations.

For research-grade segmentation, point the script at a real volumetric dataset (e.g. BraTS) and provide ground-truth masks.

## License

This project is licensed under the terms in `LICENSE`.
