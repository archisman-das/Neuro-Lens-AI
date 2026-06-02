"""Run the same v8 inference pipeline the deployed dashboard uses against
every sample under samples/ood/, then print a verdict table.

Pipeline mirrors dashboard.py:
  - load model/best_micro.onnx (ConvNeXt-Tiny U-Net, 384 px, Tversky)
  - resize -> 384, ImageNet normalise, batched 4-way flip TTA in one ORT call
  - per-pixel mean probability, threshold 0.20 -> binary mask
  - report tumor area (px), max prob, classifier verdict, image source
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ONNX = ROOT / 'model' / 'best_micro.onnx'
SAMPLES_DIR = ROOT / 'samples' / 'ood'
SIZE = 384
THRESH = 0.20
MIN_TUMOR_AREA = 50  # match dashboard's 50-pixel minimum

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_v8() -> ort.InferenceSession:
    providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(str(ONNX), providers=providers)
    print(f'[init] ONNX session: {ONNX.name} '
          f'(input={sess.get_inputs()[0].name}, output={sess.get_outputs()[0].name})')
    return sess


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL -> (3, SIZE, SIZE) float32 normalised."""
    img = img.convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)  # CHW


def tta_predict(sess: ort.InferenceSession, chw: np.ndarray) -> np.ndarray:
    """Batched 4-way TTA: id, hflip, vflip, hvflip — single ORT call.
    Returns mean tumor probability map at SIZE x SIZE.
    """
    base = chw
    h = base[:, :, ::-1].copy()
    v = base[:, ::-1, :].copy()
    hv = base[:, ::-1, ::-1].copy()
    batch = np.stack([base, h, v, hv], axis=0)  # (4, 3, SIZE, SIZE)
    in_name = sess.get_inputs()[0].name
    logits = sess.run(None, {in_name: batch})[0]  # (4, 1, SIZE, SIZE)
    if logits.shape[1] > 1:  # 2-channel models -> take fg
        logits = logits[:, 1:2]
    prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    # Undo flips before averaging.
    prob[1] = prob[1, :, :, ::-1]
    prob[2] = prob[2, :, ::-1, :]
    prob[3] = prob[3, :, ::-1, ::-1]
    return prob.mean(axis=0)[0]  # (SIZE, SIZE)


def main():
    if not ONNX.exists():
        print(f'ERROR: {ONNX} missing — download via dashboard or upload script.')
        sys.exit(2)
    sess = load_v8()
    rows: list[dict] = []
    samples = sorted(p for p in SAMPLES_DIR.rglob('*.png'))
    if not samples:
        print(f'ERROR: no PNGs under {SAMPLES_DIR}')
        sys.exit(2)
    print(f'\n[eval] {len(samples)} OOD samples\n')
    t0 = time.perf_counter()
    for p in samples:
        try:
            img = Image.open(p)
            chw = preprocess(img)
            prob = tta_predict(sess, chw)
            area = int((prob >= THRESH).sum())
            verdict = 'TUMOR' if area >= MIN_TUMOR_AREA else 'no_tumor'
            source = p.parent.name
            row = {
                'source': source,
                'file': p.name,
                'prob_max': float(prob.max()),
                'prob_mean_fg': float(prob[prob >= THRESH].mean()) if area else 0.0,
                'tumor_area_px': area,
                'verdict': verdict,
            }
            rows.append(row)
        except Exception as exc:
            print(f'  [fail] {p.name}: {type(exc).__name__}: {exc}')
    elapsed = time.perf_counter() - t0

    # Per-source summary
    print(f'\n=== per-image verdicts (threshold={THRESH}) ===')
    hdr = f'{"source":36s}  {"file":48s}  {"pmax":>5s}  {"area":>6s}  verdict'
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["source"][:36]:36s}  {r["file"][:48]:48s}  '
              f'{r["prob_max"]:.3f}  {r["tumor_area_px"]:6d}  {r["verdict"]}')

    # Aggregate per source
    print('\n=== per-source summary ===')
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    for src in sorted(by_src):
        rs = by_src[src]
        n_tum = sum(1 for r in rs if r['verdict'] == 'TUMOR')
        avg_pmax = np.mean([r['prob_max'] for r in rs])
        print(f'  {src:46s}  n={len(rs):3d}  tumor_called={n_tum:3d}  '
              f'mean(pmax)={avg_pmax:.3f}')
    print(f'\n[done] {len(rows)} samples in {elapsed:.1f}s '
          f'({elapsed/max(1,len(rows)):.2f} s/sample)')

    # Persist for inspection.
    out_csv = SAMPLES_DIR / 'eval_results.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[csv] wrote {out_csv}')


if __name__ == '__main__':
    main()
