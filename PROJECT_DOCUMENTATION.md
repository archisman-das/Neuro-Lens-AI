# NeuroLens AI — Project Documentation

## Project Overview
NeuroLens AI is a brain tumor detection project built around medical MRI image classification. It compares three model families:

- **CNN baseline** (`cnn`) — custom convolutional neural network
- **Transfer Learning** (`transfer`) — ResNet50-based classifier
- **Vision Transformer** (`vit`) — hybrid ResNet backbone + transformer classifier

The repository includes dataset utilities, training pipelines, evaluation metrics, explainability support, and a local HTML dashboard for upload-based inference.

## Repository Structure

- `README.md` — short project summary and quick start guidance
- `PROJECT_DOCUMENTATION.md` — detailed project documentation with results
- `app.py` — primary Streamlit dashboard entry point for the web UI
- `dashboard.py` — local static dashboard server for `web_dashboard/`
- `requirements.txt` — Python package dependencies
- `dataset/` — train/val/test dataset splits used for model training and evaluation
- `artifacts/` — model checkpoints, training histories, and explainability output directory
- `real_eval_current/` — latest evaluation metrics for current models
- `real_eval_fixed/` — finalized evaluation metrics after fixed model runs
- `src/` — core application logic
  - `data.py` — dataset loaders, augmentation, and preprocessing pipelines
  - `models.py` — model definitions and `get_model()` factory
  - `train.py` — training workflow and checkpoint saving
  - `evaluate.py` — model evaluation and JSON metrics exporter
  - `explain.py` — Grad-CAM and ViT saliency explainability generation
  - `predict.py` — prediction helper for saved models
  - `utils.py` — shared utilities for metrics, plotting, and data handling
- `web_dashboard/` — static dashboard files for local browser interface

## Core Design and Architecture

### Data pipeline

`src/data.py` supports two dataset formats:

- explicit split directories:
  - `dataset/train/`
  - `dataset/val/`
  - `dataset/test/`

- or a single training directory with automatic validation split if `val/` and `test/` are missing.

Dataset loading uses `tf.keras.preprocessing.image_dataset_from_directory()` to infer labels from subfolder names and resize images to `224x224`.

### Models

`src/models.py` defines three model builders:

- `build_cnn_baseline()`
  - 3 convolutional blocks, dropout, flatten, and dense classifier
- `build_transfer_model()`
  - Uses `ResNet50` or `VGG16` backbones
  - Adds dropout, dense layers, and a sigmoid output
  - Supports optional fine-tuning
- `build_vit_classifier()`
  - Uses a frozen `ResNet50` backbone as a hybrid feature extractor
  - Projects feature maps into patch tokens
  - Applies transformer blocks with multi-head attention
  - Pools and classifies with a sigmoid head

The factory `get_model(model_name, ...)` returns the requested model type for training, evaluation, or inference.

### Training workflow

`src/train.py` trains a model on the dataset and saves:

- weights to `artifacts/<model>/best_weights.weights.h5`
- history to `artifacts/<model>/history_<timestamp>.npz`
- training progress plots in the same artifact folder

Training command example:

```powershell
python src\train.py --model cnn --dataset dataset --epochs 10 --batch_size 32
```

Training uses Adam optimizer and binary crossentropy loss, plus metrics: accuracy, precision, recall.

### Evaluation workflow

`src/evaluate.py` loads a trained model, evaluates on the test split, computes:

- classification report
- confusion matrix
- ROC AUC

It saves the JSON metrics under `artifacts/<model>_evaluation_metrics.json`.

Evaluation command example:

```powershell
python src\evaluate.py --model vit --weights artifacts\vit\best_weights.weights.h5
```

### Explainability workflow

`src/explain.py` generates explainability outputs for selected models:

- Grad-CAM overlays for `cnn` and `transfer` models
- ViT patch saliency overlays for `vit`

Outputs are written to `artifacts/<model>/explain/`.

### Dashboard interface

This repository has two dashboard modes:

1. **Streamlit dashboard** — launched via `python app.py`
2. **Static web dashboard** — served by `dashboard.py` from `web_dashboard/`

The static dashboard currently provides upload-driven prediction, model comparison, and report generation.

## Results Summary

The repository contains real evaluation results in two folders:

### `real_eval_current/`

| Model    | Accuracy | Precision | Recall | F1 Score | ROC AUC | Notes |
|---------|----------|-----------|--------|----------|---------|-------|
| CNN     | 0.9750   | 0.9754    | 0.9750 | 0.9750   | 0.9928  | Current CNN evaluation on 400 samples |
| Transfer| 0.8575   | 0.8784    | 0.8575 | 0.8555   | 0.9323  | Transfer ResNet-based model (current) |
| ViT     | 0.9700   | 0.9702    | 0.9700 | 0.9700   | 0.9848  | Vision Transformer hybrid model |

### `real_eval_fixed/`

| Model    | Accuracy | Precision | Recall | F1 Score | ROC AUC | Notes |
|---------|----------|-----------|--------|----------|---------|-------|
| Transfer| 0.9825   | 0.9828    | 0.9825 | 0.9825   | 0.9956  | Fixed/resolved final evaluation |
| ViT     | 0.9800   | 0.9804    | 0.9800 | 0.9800   | 0.9975  | Fixed/resolved final evaluation |

#### Confusion matrix examples

- **CNN current:**
  - True negatives: 198
  - False positives: 2
  - False negatives: 8
  - True positives: 192
- **ViT fixed:**
  - True negatives: 199
  - False positives: 1
  - False negatives: 7
  - True positives: 193

These results show the project has strong classification performance on the available evaluation split, with Vision Transformer and Transfer models reaching very high ROC AUC values after the final fix.

## How to run the project

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare dataset

Place files in `dataset/` with class subfolders under `train/`, `val/`, and `test/`.

If your data is only in one folder, use an external split utility or modify `src/data.py` accordingly.

### 3. Train a model

```powershell
python src\train.py --model cnn --dataset dataset --epochs 10 --batch_size 32
```

### 4. Evaluate a trained model

```powershell
python src\evaluate.py --model vit --weights artifacts\vit\best_weights.weights.h5
```

### 5. Generate explainability outputs

```powershell
python src\explain.py --model transfer --weights artifacts\transfer\best_weights.weights.h5
```

### 6. Run the web dashboard

```powershell
python dashboard.py --port 8501
```

Then open:

- `http://localhost:8501/`

This launches the local HTML dashboard contained in `web_dashboard/`.

## Key observations

- The repository implements a robust comparison between CNN, transfer learning, and hybrid Vision Transformer models.
- Training and evaluation are modular and centralized in the `src/` package.
- The `dashboard.py` server provides a browser-based ML inference UI using local artifacts.
- The project has both current evaluation results and fixed final metrics, showing iterative improvement.

## Recommendations and next steps

1. **Update the Streamlit dashboard** so it reflects the latest metric files and upload flow.
2. **Add a reproducible dataset preparation script** for `dataset/raw` input, if not already available.
3. **Improve the web dashboard** with a dedicated model comparison view and downloadable PDF reports.
4. **Add consistent versioning** to artifact outputs and JSON metrics.
5. **Consolidate documentation** by referencing this `PROJECT_DOCUMENTATION.md` from `README.md`.

## Notes

- The labels are inferred from two classes: likely `tumor` and `no_tumor`.
- The current web dashboard uses `web_dashboard/index.html` and `web_dashboard/app.js`.
- Model evaluation metrics are stored as JSON and can be used for dashboard summary cards.

---

For implementation details and source-level documentation, review the files in `src/`.
