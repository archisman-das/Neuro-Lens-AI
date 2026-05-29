# NeuroLens AI - HuggingFace Spaces (Docker SDK) image.
#
# Build target: a small public demo of the layered brain-MRI tumor pipeline.
# Inference is ONNX-only (~430 MB total for v3 UNet + T1c specialist + 3
# classifiers), so CPU on free Spaces is fast enough (~30-40 ms/image).
# The LLM explanation defaults to HuggingFace Inference Providers (open-weight
# Llama 3.3 70B + Gemma 3 27B IT via $HF_TOKEN Space secret); if no token is
# set the dashboard falls back to the deterministic radiology report which is
# rich on its own and zero-hallucination by construction.
#
# Model weights are NOT bundled into this image (the HF Space free-tier 1 GB
# repo cap is too small). dashboard.py downloads them from a separate HF
# Model repo (default: Tubai01/neurolens-models) on first boot via
# huggingface_hub. See scripts/upload_models_to_hf.py for how to populate
# that Model repo from your local .pt -> .onnx exports.

FROM python:3.11-slim

# System deps: libgl/libglib for OpenCV (cv2), libgomp for ONNXruntime
# parallelism, curl for the Spaces health probe.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker layer caching speeds up rebuilds when
# code changes but requirements don't.
COPY requirements-spaces.txt /app/requirements-spaces.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-spaces.txt

# Copy application code. We intentionally do NOT copy datasets, training
# scripts, or any model weights; only what the dashboard actually needs at
# request time. ONNX weights are fetched from Tubai01/neurolens-models on
# first boot - see _ensure_onnx_models_downloaded() in dashboard.py.
COPY dashboard.py /app/dashboard.py
COPY src /app/src
COPY web_dashboard /app/web_dashboard
# Empty per-model directories so .pt/.onnx downloads land in predictable
# paths (find_weights_path searches these). Also drop in any small
# metrics .json that's present (for /metrics) - missing is fine.
COPY real_eval_current /app/real_eval_current
COPY segmentation_artifacts /app/segmentation_artifacts

# Spaces convention: PORT=7860 and bind 0.0.0.0. The dashboard reads both from
# environment so no CLI flag is needed.
ENV PORT=7860 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LOG_LEVEL=INFO

# Prefer ONNX over PyTorch on the inference path. Override with ONNX_DISABLE=1
# to debug-fall-back to the PyTorch path.
ENV ONNX_DISABLE=

# LLM defaults for the public demo. The deployer adds HF_TOKEN as a Space
# secret to enable the layered LLM pipeline; without it the dashboard returns
# the deterministic radiology report (still very rich + zero hallucinations).
ENV HF_MODEL_TEXT="meta-llama/Llama-3.3-70B-Instruct" \
    HF_MODEL_VISION="google/gemma-3-27b-it"

# Where to fetch ONNX weights from on first boot. Point at your own Model
# repo if you forked.
ENV HF_MODELS_REPO="Tubai01/neurolens-models"

EXPOSE 7860

# HF Spaces health probes /health (we expose it explicitly in dashboard.py).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:${PORT}/health || exit 1

CMD ["python", "dashboard.py"]
