import json
import os
import re
import sys
import time
import uuid
import gzip
import logging
import threading
import urllib.parse
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image
import numpy as np
import base64
import io

# ---------------------------------------------------------------------------
# Logging - one structured-ish line per request, sent to stdout (Spaces-friendly).
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s %(levelname)s %(name)s | %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger('neurolens.dashboard')

# Server version surfaced via /version and X-Server-Version header. Updated
# manually when shipping a notable change so the frontend can detect mismatches.
SERVER_VERSION = '2.1.0-onnx'

# Track when the server process started for /health uptime.
PROCESS_START_TS = time.time()
# TensorFlow is only needed for the legacy .h5 classifier branch (Grad-CAM
# via tf.expand_dims). Since all current checkpoints are PyTorch .pt, TF is
# imported lazily inside predict_image() instead of at module load. This lets
# the dashboard run on Python builds (e.g. 3.14) where TF isn't available.

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / 'web_dashboard'
MODEL_TYPES = ['cnn', 'transfer', 'vit']
MODEL_LABELS = {'cnn': 'CNN', 'transfer': 'Transfer Learning', 'vit': 'Vision Transformer'}
ARTIFACTS_DIRS = [ROOT_DIR / 'real_eval_fixed', ROOT_DIR / 'real_eval_current', ROOT_DIR / 'artifacts']
# Probe these in order; the first one with a best_model.pt wins.
#  - attention_unet_v3:  SMP UNet+ResNet34, BraTS+LGG (real masks). micro-Dice 0.910 BraTS.
#  - attention_unet_v2:  SMP UNet+ResNet34, LGG + Kaggle pseudo masks.
#  - attention_unet_lgg: hand-rolled Attention U-Net, LGG only.
#  - attention_unet:     pseudo-mask baseline (traces skull); historical reference.
SEGMENTATION_DIRS = [
    ROOT_DIR / 'segmentation_artifacts' / 'attention_unet_v3',
    ROOT_DIR / 'segmentation_artifacts' / 'attention_unet_v2',
    ROOT_DIR / 'segmentation_artifacts' / 'attention_unet_lgg',
    ROOT_DIR / 'segmentation_artifacts' / 'attention_unet',
]
# Per-modality model overrides. /segment can request a specific specialist by
# passing modality=<key> in the multipart form; we then load that model
# instead of the default search. T1c specialist is trained from the BraTS T1c
# channel triplicated as RGB (see prepare_brats_dataset.py --channels t1ce t1ce t1ce
# and segmentation_artifacts/attention_unet_t1c/).
MODALITY_DIRS = {
    't1c': ROOT_DIR / 'segmentation_artifacts' / 'attention_unet_t1c',
}
MODEL_CACHE = {}
SEG_CACHE = {}
# ONNX runtime sessions, keyed by .onnx path. Separate from the PyTorch caches
# because ONNX sessions hold GPU memory through onnxruntime, not torch's
# allocator, so we don't want them to participate in the torch cache evictor.
ONNX_CACHE: dict = {}

# Set ONNX_DISABLE=1 to force the PyTorch path (useful for Grad-CAM-heavy
# debugging or when you suspect an ONNX-vs-PyTorch numerical discrepancy on a
# new export). Default: prefer ONNX whenever a .onnx sibling exists next to
# the .pt checkpoint. Grad-CAM always falls back to PyTorch regardless of this
# flag because ONNX has no autograd.
USE_ONNX = os.environ.get('ONNX_DISABLE', '').strip() not in ('1', 'true', 'yes')

sys.path.append(str(ROOT_DIR))


def _resolve_segmentation_weights(modality: str | None = None):
    """Return (weights_path, dir_name) of the model to load.

    If `modality` matches a key in MODALITY_DIRS and that specialist has a
    best_model.pt, we use it. Otherwise fall back to the default search order
    in SEGMENTATION_DIRS (latest -> oldest).
    """
    if modality and modality in MODALITY_DIRS:
        p = MODALITY_DIRS[modality] / 'best_model.pt'
        if p.exists():
            return p, MODALITY_DIRS[modality].name
    for d in SEGMENTATION_DIRS:
        p = d / 'best_model.pt'
        if p.exists():
            return p, d.name
    return None, None


def _load_segmentation_model(modality: str | None = None):
    """Load the trained PyTorch segmentation model into the cache.

    Returns (model, device, config) on success, or None if no checkpoint
    exists for the requested modality or in the default search path.
    """
    weights_path, dir_name = _resolve_segmentation_weights(modality)
    if weights_path is None:
        return None
    cache_key = ('seg', str(weights_path), weights_path.stat().st_mtime if weights_path.exists() else 0)
    if cache_key in SEG_CACHE:
        return SEG_CACHE[cache_key]
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    cfg = ckpt.get('config', {}) or {}
    cfg['_source_dir'] = dir_name  # surface which checkpoint we loaded in /segment responses

    # v2 checkpoints record their SMP architecture + encoder; load via SMP.
    # v1 checkpoints use the hand-rolled AttentionUNet.
    architecture = ckpt.get('architecture')
    encoder = ckpt.get('encoder')
    if architecture and encoder:
        import segmentation_models_pytorch as smp
        SmpClass = getattr(smp, architecture)
        model = SmpClass(
            encoder_name=encoder,
            encoder_weights=None,  # state_dict overrides weights, don't re-download ImageNet
            in_channels=3,
            classes=1,
        ).to(device)
        cfg['_normalization'] = 'imagenet'  # tell segment_image to use ImageNet mean/std
        if 'image_size' in ckpt:
            cfg.setdefault('image_size', int(ckpt['image_size']))
    else:
        from src.segmentation_torch import AttentionUNet
        model = AttentionUNet(
            in_channels=3,
            base_filters=int(cfg.get('base_filters', 32)),
            dropout=float(cfg.get('dropout', 0.2)),
        ).to(device)
        cfg['_normalization'] = 'rescale_255'
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    # Don't clear: cascade routing wants both v3 and the T1c specialist to stay
    # GPU-warm so the fallback path doesn't pay a second 200ms reload. ~200 MB
    # combined VRAM, trivial on an 8 GB card. We cap the dict at 4 entries to
    # avoid unbounded growth if more specialists are added later.
    if len(SEG_CACHE) >= 4:
        SEG_CACHE.pop(next(iter(SEG_CACHE)))
    SEG_CACHE[cache_key] = (model, device, cfg)
    return SEG_CACHE[cache_key]


# Cascade threshold: if v3 finds fewer than this many tumor pixels on a 256x256
# slice (~0.04% of image area), treat v3 as having "found nothing" and retry
# with the T1c specialist. Empirically v3's true positives are >>500 px; sub-25
# masks are almost always background noise. Tweak in one place if you move
# image_size off 256.
CASCADE_MIN_AREA_PX = 25


def _get_onnx_session(onnx_path):
    """Return a cached onnxruntime InferenceSession for the given .onnx path.

    Sessions are reused across requests (model load is the expensive part).
    Provider preference: CUDA -> CPU. We don't enable TensorRT by default
    because its build-time graph compilation adds 30+ seconds to the first
    request, which would hurt the user-perceived 'first inference' latency.
    """
    onnx_path = str(onnx_path)
    sess = ONNX_CACHE.get(onnx_path)
    if sess is not None:
        return sess
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    providers = []
    avail = ort.get_available_providers()
    if 'CUDAExecutionProvider' in avail:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    so = ort.SessionOptions()
    # Optimization level: ALL = constant folding + fusion + memory planning.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # silence the routine memcpy / EP warnings
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    ONNX_CACHE[onnx_path] = sess
    return sess


def _segmentation_onnx_path(pt_path):
    """Return the .onnx sibling if it exists. Conventional name is the same
    basename with .onnx (produced by scripts/export_onnx.py)."""
    p = Path(pt_path)
    candidate = p.with_suffix('.onnx')
    return candidate if candidate.exists() else None


def _classifier_onnx_path(pt_path):
    """Same convention for classifier .pt files (best_weights.pt -> best_weights.onnx)."""
    p = Path(pt_path)
    candidate = p.with_suffix('.onnx')
    return candidate if candidate.exists() else None


def _segment_one(image_bytes, threshold: float, modality: str | None):
    """Single-model segmentation. No cascade logic. Returns the standard
    response dict (success/mask/overlay/tumor_area_px/source_dir) or an error
    dict if no checkpoint exists for the requested modality.

    Inference backend selection:
      - If a sibling .onnx file exists next to the resolved .pt AND USE_ONNX
        is True, run via onnxruntime (CUDA EP if available, else CPU EP).
        ~3x faster than PyTorch cold path, ~equal once warm.
      - Otherwise fall back to PyTorch. The PyTorch path is also used when
        Grad-CAM is later requested (autograd needed).
    """
    import io
    import base64
    import numpy as np
    from PIL import Image

    weights_path, dir_name = _resolve_segmentation_weights(modality)
    if weights_path is None:
        return {
            'success': False,
            'error': 'Segmentation weights not found.',
            'hint': 'Run `python src/train_segmentation_torch.py` to train the Attention U-Net first.',
        }

    onnx_path = _segmentation_onnx_path(weights_path) if USE_ONNX else None
    used_runtime = 'pytorch'

    if onnx_path is not None:
        # ONNX fast path. We don't need to load the PyTorch checkpoint at all
        # for the forward pass; we just need the cfg-style metadata (image
        # size, normalization). We default to the v2/v3 ImageNet stack since
        # all current .onnx exports came from SMP UNets with that pretraining.
        sess = _get_onnx_session(onnx_path)
        if sess is None:
            onnx_path = None  # onnxruntime missing -> fall through to PyTorch
    if onnx_path is not None:
        sess = _get_onnx_session(onnx_path)
        image_size = 256  # all current .onnx exports were taken at 256x256
        # The lgg / attention_unet baselines were trained with 0-1 rescale,
        # the SMP-style models (v2/v3/t1c) expect ImageNet normalisation.
        norm_mode = 'rescale_255' if dir_name in ('attention_unet', 'attention_unet_lgg') else 'imagenet'

        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((image_size, image_size))
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        if norm_mode == 'imagenet':
            x_np = ((arr - np.array([0.485, 0.456, 0.406], dtype=np.float32))
                    / np.array([0.229, 0.224, 0.225], dtype=np.float32))
        else:
            x_np = arr
        x_np = x_np.transpose(2, 0, 1)[None].astype(np.float32)
        logits = sess.run(None, {'input': x_np})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        probs = probs[0, 0]
        cfg = {'_source_dir': dir_name, '_normalization': norm_mode,
               'image_size': image_size}
        used_runtime = 'onnx'
    else:
        # PyTorch fallback (also takes the original loading path so cfg gets
        # populated from the checkpoint, including any custom image_size).
        import torch
        loaded = _load_segmentation_model(modality=modality)
        if loaded is None:
            return {
                'success': False,
                'error': 'Segmentation weights not found.',
                'hint': 'Run `python src/train_segmentation_torch.py` first.',
            }
        model, device, cfg = loaded
        image_size = int(cfg.get('image_size', 256))
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((image_size, image_size))
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        if cfg.get('_normalization') == 'imagenet':
            norm = arr.copy()
            norm = (norm - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
                    np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
        else:
            x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_bin = (probs >= float(threshold)).astype(np.uint8) * 255
    tumor_area_px = int((mask_bin > 0).sum())

    rgb = (arr * 255).astype(np.uint8)
    overlay = rgb.copy()
    alpha_mask = (mask_bin > 0)
    if alpha_mask.any():
        overlay[alpha_mask] = (0.4 * np.array([34, 197, 94], dtype=np.uint8) + 0.6 * overlay[alpha_mask]).astype(np.uint8)

    def _encode_png(np_img):
        buf = io.BytesIO()
        Image.fromarray(np_img).save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

    # Mean probability inside the predicted mask - useful for cascade
    # tie-breaking and for confidence display in the UI.
    if tumor_area_px > 0:
        mean_prob = float(probs[probs >= float(threshold)].mean())
    else:
        mean_prob = float(probs.max())  # what was the best we could do?

    return {
        'success': True,
        'model': 'attention_unet',
        'source_dir': cfg.get('_source_dir', 'attention_unet'),
        'runtime': used_runtime,  # 'onnx' (preferred) or 'pytorch' (fallback)
        'threshold': float(threshold),
        'image_size': image_size,
        'mask': _encode_png(mask_bin),
        'overlay': _encode_png(overlay),
        'tumor_area_px': tumor_area_px,
        'mean_prob_in_mask': mean_prob,
        'dice': None,
        'iou': None,
    }


def segment_image(image_bytes, threshold=0.5, modality: str | None = None):
    """Cascading segmentation.

    Routing rules:
      - modality='t1c' (or any other explicit key): user picked a specialist
        directly. No cascade. Return that model's output as-is. Honors the
        principle "respect the explicit user choice".
      - modality is None (default UI path): run v3 first. If v3 returns
        fewer than CASCADE_MIN_AREA_PX tumor pixels (i.e. it found nothing)
        AND the T1c specialist checkpoint exists, retry with T1c. Return
        whichever model actually found tumor. Both empty -> return v3.

    The response is augmented with a `cascade` field describing what fired:
      {
        "used": "<dir name of model returned>",
        "tried": ["attention_unet_v3", "attention_unet_t1c"],
        "primary_area_px": <int>,
        "specialist_area_px": <int>,
        "reason": "<why the cascade triggered, or 'not_triggered'>"
      }
    """
    # Explicit modality: skip cascade entirely. Honor user pick.
    if modality:
        result = _segment_one(image_bytes, threshold, modality=modality)
        if result.get('success'):
            result['cascade'] = {
                'used': result.get('source_dir'),
                'tried': [result.get('source_dir')],
                'reason': 'explicit_modality_request',
            }
        return result

    # Default path: v3 first.
    primary = _segment_one(image_bytes, threshold, modality=None)
    if not primary.get('success'):
        return primary

    primary_area = int(primary.get('tumor_area_px', 0))
    specialist_ckpt = MODALITY_DIRS.get('t1c', None)
    specialist_available = (specialist_ckpt is not None
                            and (specialist_ckpt / 'best_model.pt').exists())

    # v3 found enough tumor: no cascade needed.
    if primary_area >= CASCADE_MIN_AREA_PX or not specialist_available:
        primary['cascade'] = {
            'used': primary.get('source_dir'),
            'tried': [primary.get('source_dir')],
            'primary_area_px': primary_area,
            'specialist_area_px': None,
            'reason': ('v3_sufficient' if primary_area >= CASCADE_MIN_AREA_PX
                       else 'specialist_unavailable'),
        }
        return primary

    # v3 came up empty - try the T1c specialist.
    specialist = _segment_one(image_bytes, threshold, modality='t1c')
    specialist_area = int(specialist.get('tumor_area_px', 0)) if specialist.get('success') else 0

    if specialist_area >= CASCADE_MIN_AREA_PX:
        # Specialist rescued us - return it.
        specialist['cascade'] = {
            'used': specialist.get('source_dir'),
            'tried': [primary.get('source_dir'), specialist.get('source_dir')],
            'primary_area_px': primary_area,
            'specialist_area_px': specialist_area,
            'reason': f'v3_empty_specialist_recovered ({primary_area}px -> {specialist_area}px)',
        }
        return specialist

    # Both empty - return v3 (it's the default; specialist didn't help).
    primary['cascade'] = {
        'used': primary.get('source_dir'),
        'tried': [primary.get('source_dir'),
                  specialist.get('source_dir') if specialist.get('success') else 'attention_unet_t1c'],
        'primary_area_px': primary_area,
        'specialist_area_px': specialist_area,
        'reason': f'both_empty (v3={primary_area}px, t1c={specialist_area}px)',
    }
    return primary


def build_explanation(image_bytes, *, threshold=0.5, modality=None, backend=None,
                       modality_channels=None):
    """End-to-end pipeline behind /explain.

    Steps:
      1. /segment on the upload (PyTorch UNet, T1c specialist if modality='t1c').
      2. /predict on all 3 classifiers (cnn, transfer, vit) - reuses the cached
         models. Pulls back probability + Grad-CAM heatmap.
      3. Deterministic feature extraction via src.tumor_explainability so the
         LLM sees real numbers (area, eccentricity, GLCM, multimodal hints).
      4. LLM call via src.llm_explain. Backend selection is automatic:
         ollama -> anthropic -> openai -> deterministic local narrative.
    """
    import io as _io
    import base64 as _b64
    from PIL import Image as _PIL

    # --- 1) Segmentation ----------------------------------------------------
    seg = segment_image(image_bytes, threshold=threshold, modality=modality)
    if not seg.get('success'):
        return {'success': False, 'error': seg.get('error', 'segmentation failed'),
                'stage': 'segmentation', 'segmentation': seg}

    image_size = int(seg.get('image_size', 256))
    pil_img = _PIL.open(_io.BytesIO(image_bytes)).convert('RGB').resize((image_size, image_size))
    image_rgb = np.asarray(pil_img, dtype=np.uint8)

    def _decode_data_url(data_url):
        if not data_url:
            return None
        head, _, b64 = data_url.partition(',')
        try:
            raw = _b64.b64decode(b64)
        except Exception:
            return None
        return np.asarray(_PIL.open(_io.BytesIO(raw)).convert('RGB'), dtype=np.uint8)

    overlay_rgb = _decode_data_url(seg.get('overlay'))
    mask_rgb = _decode_data_url(seg.get('mask'))
    if mask_rgb is None:
        return {'success': False, 'error': 'segment did not return a mask',
                'stage': 'segmentation', 'segmentation': seg}
    mask_bin = (mask_rgb[..., 0] > 127).astype(np.uint8)

    # --- 2) Classifiers + Grad-CAM -----------------------------------------
    classifier_results = {}
    gradcam_for_features = None
    try:
        per_model = predict_image('all', image_bytes)
        if isinstance(per_model, dict):
            for name, res in per_model.items():
                if not isinstance(res, dict):
                    continue
                classifier_results[name] = {
                    'probability': res.get('probability'),
                    'confidence': res.get('confidence'),
                    'label': res.get('label'),
                    'display_label': res.get('display_label'),
                    'weights': res.get('weights'),
                    'gradcam': res.get('gradcam'),
                }
                # Use ViT's Grad-CAM if present (the strongest model usually)
                # else fall back to whichever has one.
                if gradcam_for_features is None and res.get('gradcam'):
                    cam_rgb = _decode_data_url(res['gradcam'])
                    if cam_rgb is not None:
                        # Convert overlay heatmap back to a [0,1] saliency proxy
                        # by taking max over channels (color intensity).
                        cam_gray = cam_rgb.max(axis=2).astype(np.float32) / 255.0
                        import cv2 as _cv2
                        gradcam_for_features = _cv2.resize(cam_gray, (image_size, image_size),
                                                           interpolation=_cv2.INTER_LINEAR)
    except Exception as exc:
        classifier_results['_error'] = f'classifier batch failed: {exc}'

    # --- 3) Deterministic feature extraction --------------------------------
    try:
        from src.tumor_explainability import extract_all_features
        features = extract_all_features(
            image_rgb=image_rgb,
            mask_bin=mask_bin,
            classifier_results=classifier_results,
            gradcam_heatmap=gradcam_for_features,
            multimodal_channels=modality_channels,
        )
    except Exception as exc:
        features = {'_error': f'feature extraction failed: {exc}'}

    # --- 4) LLM explanation -------------------------------------------------
    # Evict PyTorch models from GPU before the LLM call. Background:
    #   - empty_cache() alone only releases the *unused* cached allocator pages,
    #     not the weight tensors. With ~5 PyTorch models hot (2 UNets + 3
    #     classifiers) we permanently pin ~3 GiB, which leaves Qwen2.5-VL with
    #     too little headroom for its own weights + KV cache.
    #   - Physically dropping the cache entries -> models are garbage-collected
    #     -> empty_cache() then reclaims everything they held. The next /predict
    #     or /segment call reloads from disk (~200-500ms one-time cost) but
    #     warms the cache again. Acceptable trade for getting a 6+ GiB VL model
    #     to fit alongside our PyTorch stack on an 8 GB card.
    try:
        import gc as _gc
        import torch as _torch
        SEG_CACHE.clear()
        MODEL_CACHE.clear()
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()
    except Exception:
        pass

    try:
        from src.llm_explain import explain as llm_explain_call
        explanation = llm_explain_call(
            image_rgb=image_rgb,
            mask_bin=mask_bin,
            overlay_rgb=overlay_rgb,
            classifier_results=classifier_results,
            gradcam_rgb=_decode_data_url(seg.get('overlay')),
            features=features,
            modality_channels=modality_channels,
            backend=backend,
        )
    except Exception as exc:
        explanation = {
            'backend': 'none', 'model': '',
            'summary': f'LLM call failed ({exc}); returning deterministic features only.',
            'findings': {}, 'differential_diagnosis_hints': [],
            'model_agreement_analysis': '', 'confidence_assessment': '',
            'disclaimer': 'Not a medical diagnosis. Research / educational only.',
            'raw_features': features,
        }

    return {
        'success': True,
        'segmentation': seg,
        'classifiers': classifier_results,
        'features': features,
        'explanation': explanation,
    }


def find_weights_path(model_name):
    """Search artifact directories for a usable classifier weights file.

    Two-pass search:
      1. Any .pt in any directory wins outright. The PyTorch checkpoints from
         retrain_classifiers_torch.py are the only ones that actually load.
      2. Fall back to .h5 only if no .pt is found anywhere. The upstream .h5
         files in real_eval_fixed/ etc. are Git LFS pointer stubs (134 bytes)
         that h5py rejects with 'file signature not found'.
    """
    # Pass 1: any .pt
    for artifacts_dir in ARTIFACTS_DIRS:
        model_dir = artifacts_dir / model_name
        if not model_dir.exists():
            continue
        explicit_pt = model_dir / 'best_weights.pt'
        if explicit_pt.exists():
            return explicit_pt
        for candidate in model_dir.glob('*.pt'):
            return candidate
    # Pass 2: any .h5 (skip LFS pointer stubs that are <1 KB)
    for artifacts_dir in ARTIFACTS_DIRS:
        model_dir = artifacts_dir / model_name
        if not model_dir.exists():
            continue
        for candidate in [
            model_dir / 'best_weights.weights.h5',
            model_dir / 'best_weights.h5',
        ]:
            if candidate.exists() and candidate.stat().st_size > 1024:
                return candidate
        for candidate in model_dir.glob('*.weights.h5'):
            if candidate.stat().st_size > 1024:
                return candidate
    return None


def summarize_metrics(metrics):
    """Normalise the per-model evaluation_metrics.json into the dashboard's
    summary shape. Supports two on-disk formats:

      A. PyTorch retrainer (retrain_classifiers_torch.py):
         {"val": {"accuracy": .., "precision": .., "roc_auc": ..,
                  "confusion_matrix": {"tn": .., "fp": .., "fn": .., "tp": ..}},
          "test": {...}}    -- we prefer test if present, else val.

      B. Legacy TF evaluator (src/evaluate.py):
         {"classification_report": {"accuracy": .., "weighted avg": {...}},
          "confusion_matrix": [[tn,fp],[fn,tp]], "roc_auc": ..}
    """
    if not isinstance(metrics, dict):
        return None

    # Format A: nested under 'val'/'test'.
    if 'test' in metrics or 'val' in metrics:
        chosen = metrics.get('test') or metrics.get('val')
        if not isinstance(chosen, dict):
            return None
        cm = chosen.get('confusion_matrix')
        confusion = None
        if isinstance(cm, dict) and all(k in cm for k in ('tn', 'fp', 'fn', 'tp')):
            confusion = {k: int(cm[k]) for k in ('tn', 'fp', 'fn', 'tp')}
        return {
            'accuracy': float(chosen['accuracy']) if chosen.get('accuracy') is not None else None,
            'precision': float(chosen['precision']) if chosen.get('precision') is not None else None,
            'recall': float(chosen['recall']) if chosen.get('recall') is not None else None,
            'f1_score': float(chosen['f1']) if chosen.get('f1') is not None else None,
            'roc_auc': float(chosen['roc_auc']) if chosen.get('roc_auc') is not None else None,
            'confusion_matrix': confusion,
        }

    # Format B: legacy TF.
    report = metrics.get('classification_report', {})
    accuracy = metrics.get('accuracy')
    if isinstance(report, dict):
        accuracy = accuracy or report.get('accuracy')
        weighted = report.get('weighted avg', report.get('weighted_avg', {}))
        matrix = metrics.get('confusion_matrix')
        confusion = None
        if isinstance(matrix, list) and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix):
            confusion = {
                'tn': int(matrix[0][0]),
                'fp': int(matrix[0][1]),
                'fn': int(matrix[1][0]),
                'tp': int(matrix[1][1]),
            }
        return {
            'accuracy': float(accuracy) if accuracy is not None else None,
            'precision': float(weighted.get('precision')) if weighted.get('precision') is not None else None,
            'recall': float(weighted.get('recall')) if weighted.get('recall') is not None else None,
            'f1_score': float(weighted.get('f1-score', weighted.get('f1_score'))) if weighted.get('f1-score', weighted.get('f1_score')) is not None else None,
            'roc_auc': float(metrics.get('roc_auc')) if metrics.get('roc_auc') is not None else None,
            'confusion_matrix': confusion,
        }
    return None


def load_model_metrics():
    data = {}
    for model_name in MODEL_TYPES:
        metrics_path = next(
            (artifacts_dir / f'{model_name}_evaluation_metrics.json'
             for artifacts_dir in ARTIFACTS_DIRS
             if (artifacts_dir / f'{model_name}_evaluation_metrics.json').exists()),
            None,
        )
        model_entry = {
            'model': model_name,
            'label': MODEL_LABELS[model_name],
            'weights_found': bool(find_weights_path(model_name)),
            'metrics_found': False,
            'metrics': None,
        }
        if metrics_path and metrics_path.exists():
            try:
                with metrics_path.open('r', encoding='utf-8') as fh:
                    metrics = json.load(fh)
                model_entry['metrics'] = summarize_metrics(metrics)
                model_entry['metrics_found'] = model_entry['metrics'] is not None
            except Exception:
                model_entry['metrics_found'] = False
        data[model_name] = model_entry
    return data


def predict_image(model_name, image_bytes):
    if model_name not in MODEL_TYPES and model_name != 'all':
        raise ValueError('Unknown model selected.')

    if model_name == 'all':
        results = {}
        for name in MODEL_TYPES:
            results[name] = predict_image(name, image_bytes)
        return results

    weights_path = find_weights_path(model_name)
    if not weights_path:
        return {
            'error': 'No trained weights found for this model.',
            'hint': f'Train {MODEL_LABELS[model_name]} and save weights in artifacts/{model_name}/best_weights.weights.h5.',
        }

    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)

    cache_key = (model_name, str(weights_path), weights_path.stat().st_mtime)
    cached = MODEL_CACHE.get(cache_key)
    is_torch = weights_path.suffix == '.pt'

    if cached is None:
        MODEL_CACHE.clear()
        if is_torch:
            # PyTorch classifier path (the only one that actually works without
            # Git LFS, since the upstream .h5 files are pointer stubs).
            import torch
            from src.classifier_torch import get_classifier
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = get_classifier(model_name).to(device)
            ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
            # strict=False: the trained checkpoints from retrain_classifiers_torch.py
            # may include extra last_conv_module.* keys (duplicates of features.6.* /
            # backbone.layer4[-1].* from an earlier version of classifier_torch.py
            # that registered last_conv_module as a child Module). Now last_conv_module
            # is a @property, so those keys are 'unexpected' at load time.
            model.load_state_dict(ckpt['state_dict'], strict=False)
            model.eval()
            cached = ('torch', model, device, bool(ckpt.get('normalize_imagenet', model_name != 'cnn')))
        else:
            # Legacy TF path - kept so we can still load a real .h5 if one is
            # ever supplied (e.g. after a manual git lfs pull).
            from src.models import get_model
            if model_name == 'vit':
                model = get_model(model_name, transfer_weights='imagenet')
            else:
                model = get_model(model_name, transfer_weights=None)
            try:
                model.load_weights(str(weights_path))
            except (ValueError, OSError) as exc:
                try:
                    model.load_weights(str(weights_path), skip_mismatch=True)
                except TypeError:
                    raise exc
            cached = ('tf', model, None, False)
        MODEL_CACHE[cache_key] = cached

    backend, model, device, normalize_imagenet = cached

    # Forward pass: prefer ONNX runtime when a .onnx sibling exists.
    # ONNX gives ~3x lower latency on CUDA for the classifier head and is the
    # primary win on CPU (Spaces deployment) where PyTorch is materially slower.
    # We still need the PyTorch model on hand for the Grad-CAM step below
    # (autograd), so the model stays loaded either way.
    runtime = 'pytorch'
    onnx_path = _classifier_onnx_path(weights_path) if USE_ONNX else None
    if onnx_path is not None:
        sess = _get_onnx_session(onnx_path)
        if sess is None:
            onnx_path = None
    if onnx_path is not None:
        arr = image_array / 255.0
        if normalize_imagenet:
            arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
                  np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x_np = arr.transpose(2, 0, 1)[None].astype(np.float32)
        logits = sess.run(None, {'input': x_np})[0]
        # CNN/Transfer return shape (1,1); ViT may return (1,) - flatten safely.
        logit = float(np.asarray(logits).reshape(-1)[0])
        score = float(1.0 / (1.0 + np.exp(-logit)))
        runtime = 'onnx'
    elif backend == 'torch':
        import torch
        arr = image_array / 255.0
        if normalize_imagenet:
            arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
                  np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x).squeeze(-1)
            score = float(torch.sigmoid(logits).item())
    else:
        score = float(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0][0])
    label = 'tumor' if score >= 0.5 else 'no_tumor'
    # Prepare response payload
    result = {
        'probability': round(score, 4),
        'confidence': round(score if label == 'tumor' else 1.0 - score, 4),
        'label': label,
        'display_label': 'Tumor detected' if label == 'tumor' else 'No tumor detected',
        'weights': str(weights_path.name),
        'runtime': runtime,  # 'onnx' or 'pytorch' - which backend produced the score
    }

    # Attach original uploaded image as data URL
    try:
        buf = io.BytesIO()
        Image.fromarray(image_array.astype('uint8')).save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        result['image'] = f'data:image/png;base64,{img_b64}'
    except Exception:
        result['image'] = None

    # Grad-CAM. For the PyTorch hybrid ViT we point at the last ResNet50
    # conv block (still a Conv2d), so the same path works for all three.
    result['gradcam'] = None
    try:
        if backend == 'torch' and model_name in ('cnn', 'transfer', 'vit'):
            result['gradcam'] = _torch_gradcam_data_url(model, model_name, image_array, normalize_imagenet, device)
        elif backend == 'tf' and model_name in ('cnn', 'transfer'):
            import tensorflow as tf  # lazy: only needed for legacy .h5 path
            from src.utils import make_gradcam_heatmap, overlay_heatmap
            conv_layer = 'conv_block_3' if model_name == 'cnn' else 'conv5_block3_out'
            heatmap = make_gradcam_heatmap(tf.expand_dims(image_array, axis=0), model, conv_layer)
            overlay = overlay_heatmap(image_array.astype('uint8'), heatmap)
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format='PNG')
            result['gradcam'] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as exc:
        print(f'[dashboard] grad-cam failed for {model_name}: {exc}', flush=True)
        result['gradcam'] = None

    return result


def _torch_gradcam_data_url(model, model_name: str, image_array_0_255: np.ndarray,
                             normalize_imagenet: bool, device) -> str:
    """PyTorch Grad-CAM on the last conv module exposed by the classifier."""
    import torch
    import matplotlib.cm as cm
    target_module = getattr(model, 'last_conv_module', None)
    if target_module is None:
        return None

    arr = image_array_0_255 / 255.0
    if normalize_imagenet:
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
              np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    # Force autograd to build a graph through the frozen backbone for the
    # transfer / vit models. Without requires_grad on the input, the
    # autograd.grad call below fails with "One of the differentiated Tensors
    # does not require grad" because every parameter in the chain to the
    # captured activation is frozen.
    x.requires_grad_(True)

    # Use a forward hook to capture activations + torch.autograd.grad to
    # compute gradients w.r.t. those activations. This avoids the backward
    # hook + nn.ReLU(inplace=True) conflict that PyTorch 2.x rejects with
    # "view is being modified inplace... incorrect gradients".
    captured = {}

    def fwd_hook(_module, _inputs, output):
        captured['act'] = output  # keep autograd graph attached

    h = target_module.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x).squeeze(-1)
            grads = torch.autograd.grad(logits.sum(), captured['act'], retain_graph=False)[0]
    finally:
        h.remove()

    act = captured['act'].detach()[0]                  # (C, H, W)
    grad = grads.detach()[0]                            # (C, H, W)
    weights = grad.mean(dim=(1, 2), keepdim=True)      # (C, 1, 1)
    cam = (weights * act).sum(dim=0)                   # (H, W)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.cpu().numpy()

    # Resize CAM to 224x224
    import cv2
    cam_resized = cv2.resize(cam_np, (224, 224), interpolation=cv2.INTER_LINEAR)
    heat = (cam_resized * 255).astype(np.uint8)
    colored = (cm.get_cmap('viridis')(heat / 255.0)[..., :3] * 255).astype(np.uint8)
    overlay = (0.5 * image_array_0_255.astype(np.float32) + 0.5 * colored.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    # ---- Per-request observability hooks --------------------------------
    def setup(self):
        super().setup()
        # Request ID: trust the client's X-Request-ID if present (good for
        # tracing through a CDN / proxy), else generate one.
        self._request_id = self.headers.get('X-Request-ID') if hasattr(self, 'headers') else None
        if not self._request_id:
            self._request_id = uuid.uuid4().hex[:12]
        self._req_start = time.perf_counter()

    def _log_request(self, status: int, extra: str = ''):
        elapsed_ms = (time.perf_counter() - self._req_start) * 1000
        logger.info(
            'req_id=%s method=%s path=%s status=%d duration_ms=%.1f %s',
            getattr(self, '_request_id', '-'),
            self.command, self.path, status, elapsed_ms, extra,
        )

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/health':
            self.respond_json({'status': 'ok',
                                'uptime_seconds': round(time.time() - PROCESS_START_TS, 1),
                                'version': SERVER_VERSION}); return
        if parsed.path == '/version':
            self.respond_json({'version': SERVER_VERSION,
                                'python': sys.version.split()[0]}); return
        if parsed.path == '/status':
            self.respond_json(_get_status_snapshot()); return
        if parsed.path == '/metrics':
            self.respond_json(load_model_metrics())
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/predict':
            self.handle_predict()
            return
        if parsed.path == '/segment':
            self.handle_segment()
            return
        if parsed.path == '/explain':
            self.handle_explain()
            return
        self.send_error(404, 'Endpoint not found')

    def handle_segment(self):
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_error(400, 'Expected multipart/form-data')
            return

        boundary_match = re.search(r'boundary=(.+)', content_type)
        if not boundary_match:
            self.send_error(400, 'Missing boundary in Content-Type header')
            return

        boundary = boundary_match.group(1)
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
        boundary_bytes = boundary.encode('utf-8')

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        form = self.parse_multipart(body, boundary_bytes)

        file_item = form.get('image')
        if not file_item or 'content' not in file_item:
            self.send_error(400, 'Missing image upload')
            return

        try:
            threshold = float(form.get('threshold') or 0.5)
        except (TypeError, ValueError):
            threshold = 0.5
        modality_raw = form.get('modality')
        modality = str(modality_raw).strip().lower() if isinstance(modality_raw, str) else None

        try:
            result = segment_image(file_item['content'], threshold=threshold, modality=modality)
            self.respond_json(result)
        except Exception as exc:
            self.respond_json({'success': False, 'error': str(exc)}, status=500)

    def handle_predict(self):
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_error(400, 'Expected multipart/form-data')
            return

        boundary_match = re.search(r'boundary=(.+)', content_type)
        if not boundary_match:
            self.send_error(400, 'Missing boundary in Content-Type header')
            return

        boundary = boundary_match.group(1)
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
        boundary_bytes = boundary.encode('utf-8')

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        form = self.parse_multipart(body, boundary_bytes)

        model_name = form.get('model')
        file_item = form.get('image')
        if not model_name or not file_item or 'content' not in file_item:
            self.send_error(400, 'Missing model or image upload')
            return

        image_bytes = file_item['content']
        try:
            result = predict_image(model_name, image_bytes)
            self.respond_json({'success': True, 'result': result})
        except Exception as exc:
            self.respond_json({'success': False, 'error': str(exc)}, status=500)

    def handle_explain(self):
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_error(400, 'Expected multipart/form-data')
            return

        boundary_match = re.search(r'boundary=(.+)', content_type)
        if not boundary_match:
            self.send_error(400, 'Missing boundary in Content-Type header')
            return

        boundary = boundary_match.group(1)
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
        boundary_bytes = boundary.encode('utf-8')

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        form = self.parse_multipart(body, boundary_bytes)

        file_item = form.get('image')
        if not file_item or 'content' not in file_item:
            self.send_error(400, 'Missing image upload')
            return

        try:
            threshold = float(form.get('threshold') or 0.5)
        except (TypeError, ValueError):
            threshold = 0.5

        modality_raw = form.get('modality')
        modality = str(modality_raw).strip().lower() if isinstance(modality_raw, str) and modality_raw else None
        backend_raw = form.get('backend')
        backend = str(backend_raw).strip().lower() if isinstance(backend_raw, str) and backend_raw else None
        if backend in ('', 'auto'):
            backend = None

        # modality_channels: optional channel triplet hint for multimodal stacks.
        # The web UI doesn't expose this yet; an API caller can pass
        # modality_channels="t1c,t2,flair" (comma-separated, 3 names).
        modality_channels = None
        mc_raw = form.get('modality_channels')
        if isinstance(mc_raw, str) and mc_raw:
            parts = [p.strip() for p in mc_raw.split(',') if p.strip()]
            if len(parts) == 3:
                modality_channels = tuple(parts)

        try:
            result = build_explanation(
                file_item['content'],
                threshold=threshold,
                modality=modality,
                backend=backend,
                modality_channels=modality_channels,
            )
            self.respond_json(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.respond_json({'success': False, 'error': str(exc)}, status=500)

    def parse_multipart(self, body, boundary):
        parts = body.split(b'--' + boundary)
        data = {}
        for part in parts:
            if not part or part in (b'--', b'--\r\n'):
                continue
            part = part.strip(b'\r\n')
            if not part:
                continue

            header_bytes, _, content = part.partition(b'\r\n\r\n')
            headers = {}
            for line in header_bytes.split(b'\r\n'):
                name, _, value = line.decode('utf-8', 'ignore').partition(':')
                headers[name.lower().strip()] = value.strip()

            disposition = headers.get('content-disposition', '')
            disposition_data = self.parse_content_disposition(disposition)
            name = disposition_data.get('name')
            if not name:
                continue

            if 'filename' in disposition_data:
                data[name] = {
                    'filename': disposition_data.get('filename'),
                    'content': content.rstrip(b'\r\n'),
                }
            else:
                data[name] = content.decode('utf-8', errors='replace').strip()
        return data

    def parse_content_disposition(self, disposition):
        values = {}
        parts = [part.strip() for part in disposition.split(';') if part.strip()]
        for part in parts:
            if '=' in part:
                key, val = part.split('=', 1)
                values[key.strip().lower()] = val.strip('"')
        return values

    def respond_json(self, data, status=200):
        payload = json.dumps(data).encode('utf-8')
        # gzip for non-trivial payloads when the client supports it. Saves
        # 60-80% on bandwidth for /explain (which carries base64-PNG dumps).
        accept_enc = self.headers.get('Accept-Encoding', '')
        gzipped = False
        if len(payload) > 1024 and 'gzip' in accept_enc.lower():
            payload = gzip.compress(payload)
            gzipped = True

        elapsed_ms = (time.perf_counter() - self._req_start) * 1000

        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        if gzipped:
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('X-Request-ID', self._request_id)
        self.send_header('X-Server-Version', SERVER_VERSION)
        self.send_header('X-Inference-Time-ms', f'{elapsed_ms:.1f}')
        # CORS - allow the dashboard hosted on Spaces to talk to itself across
        # any prefix the platform proxies through.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(payload)
        self._log_request(status, extra=f'gzipped={gzipped} size_kb={len(payload)/1024:.1f}')

    def log_message(self, format, *args):
        # Suppress the default per-line stderr from BaseHTTPRequestHandler;
        # we emit our own structured logs in respond_json / handlers.
        return


def _get_status_snapshot() -> dict:
    """Real /status endpoint: what models are actually loaded, real GPU memory,
    classifier weight presence. Replaces the previous mock UI 'System Status'
    block (which showed hard-coded '3/3 models, 4.2/8 GB GPU, 2 pending').
    """
    snap: dict = {
        'version': SERVER_VERSION,
        'uptime_seconds': round(time.time() - PROCESS_START_TS, 1),
        'classifiers': {},
        'segmentation_models': [],
        'onnx_runtime': {'available': False, 'providers': [], 'sessions_loaded': 0},
        'gpu': {'available': False},
        'llm': {
            'ollama_text_model': os.environ.get('OLLAMA_MODEL_TEXT', 'qwen2.5:1.5b'),
            'ollama_vision_model': os.environ.get('OLLAMA_MODEL_VISION', 'qwen2.5vl:3b'),
            'hf_inference_token_present': bool(os.environ.get('HF_TOKEN')),
            'anthropic_token_present': bool(os.environ.get('ANTHROPIC_API_KEY')),
        },
    }
    # Classifier weight presence (and which runtime would be used).
    for m in MODEL_TYPES:
        pt = find_weights_path(m)
        if pt:
            onnx = _classifier_onnx_path(pt) if pt.suffix == '.pt' else None
            snap['classifiers'][m] = {
                'pt': str(pt.name), 'pt_size_mb': round(pt.stat().st_size / 1e6, 1),
                'onnx': onnx.name if onnx else None,
                'preferred_runtime': 'onnx' if (onnx and USE_ONNX) else 'pytorch',
            }
        else:
            snap['classifiers'][m] = {'pt': None, 'preferred_runtime': None}
    # Segmentation model directories (which exist and which have ONNX siblings).
    for d in SEGMENTATION_DIRS + list(MODALITY_DIRS.values()):
        pt = d / 'best_model.pt'
        if pt.exists():
            onnx = _segmentation_onnx_path(pt)
            snap['segmentation_models'].append({
                'dir': d.name,
                'pt_size_mb': round(pt.stat().st_size / 1e6, 1),
                'onnx': onnx.name if onnx else None,
                'preferred_runtime': 'onnx' if (onnx and USE_ONNX) else 'pytorch',
            })
    # ONNX runtime telemetry.
    try:
        import onnxruntime as ort
        snap['onnx_runtime'] = {
            'available': True,
            'version': ort.__version__,
            'providers': ort.get_available_providers(),
            'sessions_loaded': len(ONNX_CACHE),
        }
    except ImportError:
        pass
    # GPU memory (PyTorch path).
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            snap['gpu'] = {
                'available': True,
                'name': torch.cuda.get_device_name(0),
                'memory_used_mb': round((total - free) / 1e6, 1),
                'memory_total_mb': round(total / 1e6, 1),
                'memory_free_mb': round(free / 1e6, 1),
            }
    except Exception:
        pass
    return snap


def _warm_models_async():
    """Pre-load ONNX sessions for the cascade pair (v3 + T1c) and the 3
    classifiers in a background thread so the first /predict and /segment
    requests don't pay the cold-start tax (~200-500 ms each).

    Failure is silent: if a model file isn't there or onnxruntime can't open
    it, we just log and move on - the request path will surface a real error
    if needed.
    """
    def _warm():
        t0 = time.perf_counter()
        warmed = 0
        # Classifiers
        for m in MODEL_TYPES:
            pt = find_weights_path(m)
            if pt:
                onnx = _classifier_onnx_path(pt)
                if onnx and USE_ONNX:
                    if _get_onnx_session(onnx) is not None:
                        warmed += 1
        # Segmentation cascade pair
        for d in [ROOT_DIR / 'segmentation_artifacts' / 'attention_unet_v3',
                   MODALITY_DIRS.get('t1c')]:
            if d is None:
                continue
            pt = d / 'best_model.pt'
            if pt.exists():
                onnx = _segmentation_onnx_path(pt)
                if onnx and USE_ONNX:
                    if _get_onnx_session(onnx) is not None:
                        warmed += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info('model_warmup_complete sessions=%d duration_ms=%.1f', warmed, elapsed_ms)

    threading.Thread(target=_warm, name='neurolens-warmup', daemon=True).start()


def run(port=8501, host: str = ''):
    if not WEB_DIR.exists():
        raise FileNotFoundError(f'Web dashboard files not found: {WEB_DIR}')

    address = (host, port)
    # ThreadingHTTPServer: parallel requests don't queue behind each other.
    # Important when a slow /explain LLM call would otherwise block /predict
    # or /health probes from the Spaces orchestrator.
    server = ThreadingHTTPServer(address, DashboardHandler)
    url = f'http://localhost:{port}/' if not host else f'http://{host}:{port}/'
    logger.info('neurolens_dashboard_starting version=%s url=%s', SERVER_VERSION, url)
    _warm_models_async()
    print(f'NeuroLens AI dashboard running at {url}')
    print(f'Version: {SERVER_VERSION}.  Endpoints: /predict /segment /explain '
          '/metrics /status /health /version.')
    print('Press Ctrl+C here to stop the server.')
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the NeuroLens AI HTML dashboard')
    # Defaults are env-driven so the Spaces Dockerfile can override without
    # touching the CLI. HF Spaces sets PORT=7860 on Docker SDK and expects
    # the container to bind 0.0.0.0; local dev uses 8501 on localhost.
    parser.add_argument('--port', type=int,
                         default=int(os.environ.get('PORT', '8501')))
    parser.add_argument('--host', type=str,
                         default=os.environ.get('HOST', ''))
    args = parser.parse_args()
    run(args.port, host=args.host)
