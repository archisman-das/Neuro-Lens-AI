---
title: NeuroLens AI - Brain Tumor Detection
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: ONNX-fast brain MRI segmentation, classification, and open-source LLM radiology report
tags:
  - medical-imaging
  - brain-tumor
  - segmentation
  - vision-language
  - llama
  - onnx
license: mit
---

# NeuroLens AI

Research-grade brain-MRI tumor detection demo:

- **3 PyTorch classifiers** (CNN / Transfer ResNet50 / ViT-hybrid) - **exported to ONNX**, ~30 ms/image on CPU
- **Cascade Attention U-Net segmentation** (BraTS+LGG v3 + T1c specialist fallback)
- **Layered LLM explanation** (Llama 3.3 70B for prose + Llama 3.2 90B Vision for visual co-observer) with hallucination guards: every LLM claim is citation-checked against measured features; visual claims that contradict measurements are surfaced as `disagreements`, not findings
- **Zero-hallucination deterministic radiology report** if no LLM is configured

## How to enable the LLM explanation

The dashboard ships in deterministic-only mode by default (no LLM needed). To turn on the layered LLM pipeline, add this **Space secret** in Settings -> Variables and secrets:

| Secret | Value |
|---|---|
| `HF_TOKEN` | Your HuggingFace token with **inference** scope (free) |

That's it. The dashboard auto-detects the token at startup and routes Patterns A/B to `meta-llama/Llama-3.3-70B-Instruct` and Pattern C to `meta-llama/Llama-3.2-90B-Vision-Instruct` via HF Inference Providers. All open-weight models.

Override model choice with `HF_MODEL_TEXT` / `HF_MODEL_VISION` env vars.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web dashboard UI |
| POST | `/segment` | Cascade UNet, returns mask + overlay (PNG data URLs) |
| POST | `/predict` | CNN / Transfer / ViT probability + Grad-CAM |
| POST | `/explain` | Full layered pipeline (segmentation + classifier + features + LLM) |
| GET | `/metrics` | Persisted accuracy / ROC AUC for each classifier |
| GET | `/status` | Live: loaded ONNX sessions, GPU memory, LLM backend availability |
| GET | `/health` | Liveness probe (uptime + version) |
| GET | `/version` | Server version string |

## License

Apache 2.0 for the code. Model weights inherit their original licenses (BraTS dataset terms apply for the segmentation models). Not a medical device. Research / educational use only.
