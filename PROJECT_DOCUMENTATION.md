# NeuroLens AI Project Documentation

## 1. Project Overview

NeuroLens AI is a brain MRI analysis project focused on tumor detection, model comparison, explainability, and segmentation research. The repository combines a browser-based dashboard, a Streamlit interface, TensorFlow model training scripts, evaluation utilities, Grad-CAM explainability, and advanced segmentation experiments.

The project is intended for research, learning, and prototype development. It is not a certified clinical diagnostic system and should not be used as the only basis for medical decisions.

## 2. Objectives

- Detect whether an uploaded brain MRI indicates tumor presence.
- Compare multiple classification model families: CNN, transfer learning, and Vision Transformer.
- Provide interpretable outputs through model metrics and Grad-CAM visualizations.
- Support segmentation experiments using U-Net style architectures.
- Provide reproducible training, evaluation, k-fold validation, ablation, and robustness workflows.

## 3. Repository Structure

```text
.
|-- app.py                    # Streamlit dashboard entry point
|-- dashboard.py              # Local HTTP server for the HTML dashboard
|-- web_dashboard/            # Frontend dashboard files
|-- src/                      # Core ML, training, evaluation, and segmentation code
|-- config.yaml               # Reference configuration for segmentation experiments
|-- requirements.txt          # Python package requirements
|-- README.md                 # Quick-start project overview
|-- LICENSE                   # MIT license
`-- Documentation/            # Additional report assets
```

## 4. Main Components

### 4.1 HTML Dashboard

The HTML dashboard lives in `web_dashboard/` and is served by `dashboard.py`. It provides a polished interface for image upload, model selection, metrics display, and prediction results.

Key files:

- `web_dashboard/index.html`
- `web_dashboard/style.css`
- `web_dashboard/app.js`
- `dashboard.py`

The dashboard backend exposes local endpoints for model metadata and image prediction. It looks for model weights and metrics in these folders:

- `real_eval_fixed/`
- `real_eval_current/`
- `artifacts/`

Expected classification weight path:

```text
artifacts/<model_name>/best_weights.weights.h5
```

where `<model_name>` is `cnn`, `transfer`, or `vit`.

### 4.2 Streamlit App

`app.py` provides an alternate Streamlit dashboard for model comparison and upload-driven prediction. It uses the same broad model families and artifact structure as the HTML dashboard.

### 4.3 Classification Models

Classification models are defined in `src/models.py` and trained through `src/train.py`.

Supported model choices:

- `cnn`: custom convolutional neural network
- `transfer`: transfer learning model
- `vit`: Vision Transformer model

Training output is saved under `artifacts/<model_name>/`.

### 4.4 Explainability

Explainability utilities live in `src/explain.py` and `src/utils.py`. The dashboard can generate Grad-CAM overlays for supported convolutional models when trained weights are available.

### 4.5 Segmentation Research

Segmentation code supports U-Net style architectures and advanced experiment workflows.

Important files:

- `src/segmentation_models.py`
- `src/train_segmentation.py`
- `src/advanced_models.py`
- `src/advanced_training.py`
- `src/kfold_validation.py`
- `src/ablation_study.py`
- `src/robustness_analysis.py`

Supported segmentation model variants include:

- U-Net
- Attention U-Net
- Residual U-Net
- Multi-modal U-Net

## 5. Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 6. Running the Dashboards

### 6.1 HTML Dashboard

```bash
python dashboard.py --port 8501
```

Open the dashboard at:

```text
http://localhost:8501/
```

### 6.2 Streamlit Dashboard

```bash
streamlit run app.py
```

## 7. Training Workflows

### 7.1 Classification Training

Example:

```bash
python src/train.py --model cnn --dataset dataset --epochs 10 --batch_size 32 --output artifacts
```

Common options:

- `--model`: choose `cnn`, `transfer`, or `vit`
- `--dataset`: dataset directory
- `--epochs`: number of training epochs
- `--batch_size`: batch size
- `--learning_rate`: optimizer learning rate
- `--output`: artifact output directory

### 7.2 Segmentation Training

Example:

```bash
python src/train_segmentation.py --data_dir dataset --model_type unet --epochs 100 --batch_size 16
```

Useful segmentation options:

- `--model_type`: `unet`, `attention_unet`, `res_unet`, or `multi_modal_unet`
- `--image_size`: input image size
- `--base_filters`: base convolution filter count
- `--dropout_rate`: dropout rate
- `--use_attention`: enable attention in supported models
- `--use_kfold`: run k-fold validation
- `--use_ablation`: run ablation experiments
- `--save_dir`: output directory for segmentation artifacts

## 8. Evaluation

Classification evaluation utilities are available in `src/evaluate.py`. Segmentation workflows include Dice coefficient, IoU, accuracy, precision, recall, F1 score, specificity, and threshold-based analysis where supported.

Metrics can be stored as JSON files and consumed by the dashboards for comparison views.

## 9. Dataset Expectations

The project expects local image datasets such as `dataset/`, `dataset_real/`, or source data folders. Dataset folders may be large and are usually best managed outside normal source-control workflows unless the project intentionally includes sample data.

For classification, datasets should be organized so TensorFlow data loaders can infer labels from folder structure.

For segmentation, image-mask pairing should follow the assumptions in `src/train_segmentation.py` and related preprocessing utilities.

## 10. Artifact Management

Recommended generated-output locations:

- `artifacts/` for classification weights, histories, plots, and metrics
- `real_eval_current/` and `real_eval_fixed/` for selected evaluation snapshots
- `segmentation_models/` or a configured output folder for segmentation runs

Large generated files should be reviewed before committing to GitHub.

## 11. Limitations

- Model quality depends heavily on dataset quality, labeling, preprocessing, and validation strategy.
- MRI datasets can vary across scanners, protocols, institutions, and patient populations.
- Grad-CAM highlights model attention, not guaranteed clinical causality.
- Prototype dashboards do not replace clinical workflows, radiologist review, or regulatory validation.

## 12. Future Improvements

- Add automated tests for data loading, prediction endpoints, and model utility functions.
- Add a smaller sample dataset or mock artifacts for reproducible demos.
- Add CI checks for formatting and basic import validation.
- Improve artifact versioning and experiment tracking.
- Add clearer dataset preparation documentation for classification and segmentation tasks.

## 13. License

NeuroLens AI is released under the MIT License. See `LICENSE` for details.
