"""Generate v9b JEPA anomaly-map overlays for visual localization check.

Picks ~8 OOD samples across all 4 sources (mix of tumor + healthy),
runs JEPA prediction_error_map, and saves a 3-panel figure per sample:
  [original MRI]  [anomaly heatmap]  [overlay (original + thresholded mask)]

Outputs to samples/ood/v9b_localization/*.png so you can eyeball whether
JEPA fires on the actual tumor location or just on random texture.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.jepa import IJEPAModel

JEPA_CKPT = ROOT / 'v9b_artifacts' / 'v9b_jepa' / 'last.pt'
SAMPLES = ROOT / 'samples' / 'ood'
OUT_DIR = SAMPLES / 'v9b_localization'
IMAGE_SIZE = 256
# Threshold for binary anomaly mask: matches the best-F1 operating point
# (per scripts/analyze_v9b_thresholds.py)
ANOMALY_THR = 0.40


def viridis_rgb(g: np.ndarray) -> np.ndarray:
    """Lightweight 5-anchor viridis colormap, numpy only."""
    g = np.clip(g.astype(np.float32), 0.0, 1.0)
    anchors = np.array([
        [0.267, 0.005, 0.329], [0.282, 0.140, 0.458],
        [0.254, 0.265, 0.530], [0.207, 0.372, 0.553],
        [0.993, 0.906, 0.144],
    ], dtype=np.float32)
    t = g * 4.0
    lo = np.clip(np.floor(t).astype(np.int32), 0, 3)
    hi = np.clip(lo + 1, 0, 4)
    frac = (t - lo)[..., None]
    out = anchors[lo] * (1.0 - frac) + anchors[hi] * frac
    return (out * 255).astype(np.uint8)


def load_jepa(device):
    ck = torch.load(str(JEPA_CKPT), map_location=device, weights_only=False)
    a = ck.get('args', {})
    m = IJEPAModel(image_size=a.get('image_size', 256), patch_size=16,
                    embed_dim=a.get('embed_dim', 384), depth=a.get('depth', 12),
                    heads=a.get('heads', 6))
    m.load_state_dict(ck['model_state_dict'])
    return m.to(device).eval()


def make_panel(orig_rgb: np.ndarray, emap: np.ndarray, mask: np.ndarray,
                title: str) -> np.ndarray:
    """Build a 3-panel image: original | heatmap | overlay. Returns RGB uint8."""
    H, W = orig_rgb.shape[:2]
    # Normalise heatmap to [0,1] across this single image for visualisation
    emap_norm = (emap - emap.min()) / max(emap.max() - emap.min(), 1e-6)
    heatmap_rgb = viridis_rgb(emap_norm)

    overlay = orig_rgb.copy().astype(np.float32)
    red = np.array([220, 30, 30], dtype=np.float32)
    alpha = 0.5
    overlay[mask > 0] = (1 - alpha) * overlay[mask > 0] + alpha * red
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Stitch horizontally with 4 px gap
    gap = 4
    pad = np.zeros((H, gap, 3), dtype=np.uint8) + 50
    stitched = np.concatenate([orig_rgb, pad, heatmap_rgb, pad, overlay], axis=1)
    # Add a small title bar
    bar_h = 26
    bar = np.zeros((bar_h, stitched.shape[1], 3), dtype=np.uint8) + 24
    # Crude title via PIL since we don't want a matplotlib dep
    from PIL import ImageDraw, ImageFont
    bar_pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(bar_pil)
    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 4), title, fill=(230, 230, 230), font=font)
    bar = np.array(bar_pil)
    return np.concatenate([bar, stitched], axis=0)


def main():
    if not JEPA_CKPT.exists():
        sys.exit(f'missing {JEPA_CKPT}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')
    model = load_jepa(device)

    # Pick 2 samples per source (8 total). Stratified by GT.
    picks: list[tuple[str, str, Path]] = []
    by_src: dict[str, list[Path]] = {}
    for p in sorted(SAMPLES.rglob('*')):
        if p.suffix.lower() not in ('.png','.jpg','.jpeg'): continue
        if p.parent.name not in (
            'healthy_coronal_T1_openneuro',
            'tumor_proprietary_multimodal_unidata',
            'tumor_multi_patient_ultralytics',
            'tumor_binary_navoneel_via_miladfa7',
        ):
            continue
        by_src.setdefault(p.parent.name, []).append(p)
    for src, files in by_src.items():
        # Pick first + middle to get variety
        for p in (files[0], files[len(files)//2]):
            picks.append((src, p.name, p))
    print(f'[init] {len(picks)} samples picked for localization viz')

    for src, fname, p in picks:
        gt = 'TUMOR-GT' if 'tumor' in src else 'HEALTHY-GT'
        img = Image.open(p).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            emap = model.prediction_error_map(x).squeeze().cpu().numpy()
        p95 = float(np.percentile(emap, 95))
        # Per-image threshold = take the top X% of pixels as anomalous
        # (more useful visually than the absolute scaled threshold)
        thr = np.percentile(emap, 90)
        mask = (emap > thr).astype(np.uint8)
        ano_frac = float(mask.mean())
        orig_rgb = (arr * 255).astype(np.uint8)
        title = (f'{gt}  |  src={src[:35]}  |  file={fname[:30]}  |  '
                  f'p95={p95:.3f}  anomaly_pixels={ano_frac:.0%}  '
                  f'inference={time.perf_counter()-t0:.1f}s')
        panel = make_panel(orig_rgb, emap, mask, title)
        out = OUT_DIR / f'{src}__{fname.rsplit(".",1)[0]}.png'
        Image.fromarray(panel).save(out)
        print(f'  {out.name}  (p95={p95:.3f}, gt={gt})')

    print(f'\n[done] {len(picks)} panels in {OUT_DIR}/')
    print('Open the PNGs to see whether the anomaly heatmap lights up '
          'where the tumor actually is.')


if __name__ == '__main__':
    main()
