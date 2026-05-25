# NeuroLens AI

NeuroLens AI is a brain MRI analysis project for tumor detection, model comparison, and segmentation research. It includes a browser dashboard, a Streamlit interface, TensorFlow training scripts, evaluation utilities, and research workflows for U-Net based segmentation experiments.

> This project is for research and educational use only. It is not a medical device and should not be used as the sole basis for clinical decisions.

## Features

- MRI upload and inference workflow for tumor / no-tumor classification.
- Dashboard metrics for CNN, transfer learning, and Vision Transformer models.
- Grad-CAM support for convolutional models when trained weights are available.
- Segmentation model implementations including U-Net, Attention U-Net, Res U-Net, and multi-modal U-Net variants.
- Advanced experiment scripts for k-fold validation, ablation studies, robustness analysis, and segmentation training.
- Documentation and presentation assets under `Documentation/` and `ppt/`.

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

## Train Segmentation Models

Segmentation settings are documented in `config.yaml`; the training script accepts matching command-line options.

```bash
python src/train_segmentation.py --data_dir dataset --model_type unet --epochs 100 --batch_size 16
```

The segmentation workflow supports U-Net style models, cross-validation, ablation studies, and robustness evaluation.

## Dataset Notes

The training scripts expect image datasets in local folders such as `dataset/` or `dataset_real/`. These folders are intentionally not required for simply viewing the source code or running the dashboard shell. Add trained model artifacts locally before using live inference.

## License

This project is licensed under the terms in `LICENSE`.
