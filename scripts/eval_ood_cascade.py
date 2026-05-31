"""Production-cascade eval on OOD samples.

Mirrors dashboard.py exactly (faithful to dashboard.py:902-968 and
src/llm_explain.py:_classifier_consensus):

  1. Run v8 seg + 4-way batched TTA (same as eval_ood_samples.py)
  2. Run all 3 classifiers (cnn/transfer/vit) at 224 px in one batched call
       - cnn:        /255 only
       - transfer:   /255, then ImageNet normalise
       - vit:        /255, then ImageNet normalise
  3. Compute consensus per src/llm_explain.py:1673-1701:
       tumor:    mean_p >= 0.7 AND all 3 >= 0.5
       no_tumor: mean_p <= 0.3 AND all 3 <= 0.5
       mixed:    else
       band:     high if mean_p >= 0.9 or <= 0.1; moderate otherwise
  4. Apply gating: if verdict == no_tumor and band in (high, moderate),
     mask is suppressed -> production verdict = no_tumor regardless of seg.

Also runs a threshold sweep [0.10..0.60] reusing the cached probs (no
extra inference cost) and reports per-source per-threshold FP/recall.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SEG_ONNX = ROOT / 'model' / 'best_micro.onnx'
CLF_ONNX = {
    'cnn':      ROOT / 'real_eval_current' / 'cnn'      / 'best_weights.onnx',
    'transfer': ROOT / 'real_eval_current' / 'transfer' / 'best_weights.onnx',
    'vit':      ROOT / 'real_eval_current' / 'vit'      / 'best_weights.onnx',
}
# CNN is the only one NOT ImageNet-normalized (dashboard.py:1262 + 1228).
NORMALIZE_IMAGENET = {'cnn': False, 'transfer': True, 'vit': True}

SAMPLES_DIR = ROOT / 'samples' / 'ood'
SEG_SIZE = 384
CLF_SIZE = 224
DEFAULT_THRESHOLD = 0.20
MIN_TUMOR_AREA = 50

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Source -> ground-truth label.
GT = {
    'healthy_coronal_T1_openneuro':           'no_tumor',
    'tumor_proprietary_multimodal_unidata':   'tumor',
    'tumor_multi_patient_ultralytics':        'tumor',  # 10 distinct patients
    'tumor_binary_navoneel_via_miladfa7':     'tumor',  # ~6 patients, binary set
}


def _sess(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])


def _preprocess_seg(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IM_MEAN) / IM_STD
    return arr.transpose(2, 0, 1)


def _preprocess_clf(img: Image.Image, normalise: bool) -> np.ndarray:
    img = img.convert('RGB').resize((CLF_SIZE, CLF_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if normalise:
        arr = (arr - IM_MEAN) / IM_STD
    return arr.transpose(2, 0, 1)


def seg_tta(sess: ort.InferenceSession, chw: np.ndarray) -> np.ndarray:
    h = chw[:, :, ::-1].copy()
    v = chw[:, ::-1, :].copy()
    hv = chw[:, ::-1, ::-1].copy()
    batch = np.stack([chw, h, v, hv], axis=0)
    logits = sess.run(None, {sess.get_inputs()[0].name: batch})[0]
    if logits.shape[1] > 1:
        logits = logits[:, 1:2]
    prob = 1.0 / (1.0 + np.exp(-logits))
    prob[1] = prob[1, :, :, ::-1]
    prob[2] = prob[2, :, ::-1, :]
    prob[3] = prob[3, :, ::-1, ::-1]
    return prob.mean(axis=0)[0]


def classify_all(clfs: dict, img: Image.Image) -> dict:
    """Return per-classifier tumor probabilities."""
    out = {}
    for name, sess in clfs.items():
        chw = _preprocess_clf(img, NORMALIZE_IMAGENET[name])
        logit = float(sess.run(None, {sess.get_inputs()[0].name: chw[None]})[0].reshape(-1)[0])
        out[name] = 1.0 / (1.0 + np.exp(-logit))
    return out


def consensus(probs: dict) -> tuple:
    """Mirror src/llm_explain.py:_classifier_consensus."""
    vs = [p for p in probs.values() if isinstance(p, (int, float))]
    if not vs:
        return None, None, None
    mean_p = sum(vs) / len(vs)
    all_above = all(p >= 0.5 for p in vs)
    all_below = all(p <= 0.5 for p in vs)
    if mean_p >= 0.7 and all_above:
        band = 'high' if mean_p >= 0.9 else 'moderate'
        return 'tumor', mean_p, band
    if mean_p <= 0.3 and all_below:
        band = 'high' if mean_p <= 0.1 else 'moderate'
        return 'no_tumor', mean_p, band
    return 'mixed', mean_p, 'low'


# Pull a modality hint from UniData filenames like "SE000009__T2W_FLAIRanonymized__..."
MODALITY_RE = re.compile(r'(T1W?[_ ]?(?:FFE|SAG|Cor|TSE)?|T2W?[_ ]?(?:TSE|COR|FLAIR)?|FLAIR|DWI|Survey|MPRAGE)',
                         re.IGNORECASE)


def modality_of(filename: str) -> str:
    m = MODALITY_RE.search(filename)
    if not m:
        return 'unknown'
    raw = m.group(0).upper().replace(' ', '_')
    # Coalesce variants
    if 'FLAIR' in raw: return 'FLAIR'
    if 'DWI' in raw: return 'DWI'
    if 'SURVEY' in raw: return 'survey'
    if 'T2' in raw and 'COR' in raw: return 'T2_coronal'
    if 'T2' in raw: return 'T2_axial'
    if 'T1' in raw and 'SAG' in raw: return 'T1_sagittal'
    if 'T1' in raw and 'COR' in raw: return 'T1_coronal'
    if 'T1' in raw and 'FFE' in raw: return 'T1_axial_FFE'
    if 'T1' in raw: return 'T1'
    return raw


def main():
    if not SEG_ONNX.exists():
        sys.exit(f'missing {SEG_ONNX}')
    for n, p in CLF_ONNX.items():
        if not p.exists():
            sys.exit(f'missing {n}: {p}')

    seg = _sess(SEG_ONNX)
    clfs = {n: _sess(p) for n, p in CLF_ONNX.items()}
    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not samples:
        sys.exit(f'no PNGs under {SAMPLES_DIR}')
    print(f'[init] seg=v8 + clf={{cnn,transfer,vit}} | samples={len(samples)}')

    t0 = time.perf_counter()
    rows = []
    for p in samples:
        img = Image.open(p)
        probs = classify_all(clfs, img)
        verdict, mean_p, band = consensus(probs)
        prob_map = seg_tta(seg, _preprocess_seg(img))
        area_at_default = int((prob_map >= DEFAULT_THRESHOLD).sum())
        seg_says_tumor = area_at_default >= MIN_TUMOR_AREA
        # Production gating: classifier consensus suppresses the mask.
        mask_suppressed = (verdict == 'no_tumor' and band in ('high', 'moderate'))
        if mask_suppressed:
            cascade_verdict = 'no_tumor'
        elif verdict == 'tumor' and band in ('high', 'moderate'):
            cascade_verdict = 'TUMOR'
        elif seg_says_tumor:
            cascade_verdict = 'TUMOR'  # mixed-classifier + non-empty seg -> tumor
        else:
            cascade_verdict = 'no_tumor'
        rows.append({
            'source': p.parent.name,
            'file': p.name,
            'modality': modality_of(p.name),
            'gt': GT.get(p.parent.name, 'unknown'),
            'p_cnn': round(probs['cnn'], 3),
            'p_transfer': round(probs['transfer'], 3),
            'p_vit': round(probs['vit'], 3),
            'mean_p': round(mean_p, 3) if mean_p else None,
            'clf_verdict': verdict,
            'clf_band': band,
            'mask_suppressed': mask_suppressed,
            'v8_only_verdict': 'TUMOR' if seg_says_tumor else 'no_tumor',
            'cascade_verdict': cascade_verdict,
            'prob_map': prob_map,  # kept for threshold sweep
            'tumor_area_at_0.20': area_at_default,
        })
    elapsed = time.perf_counter() - t0
    print(f'[done] {len(rows)} samples in {elapsed:.1f}s ({elapsed/len(rows):.2f}/sample)')

    # ---- Per-image cascade table -----------------------------------------
    print('\n=== per-image cascade decisions ===')
    hdr = f'{"src":36s} {"file":48s} {"mod":15s} cnn  tr   vit  mean band     v8only cascade GT'
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f'{r["source"][:36]:36s} {r["file"][:48]:48s} {r["modality"][:15]:15s} '
              f'{r["p_cnn"]:.2f} {r["p_transfer"]:.2f} {r["p_vit"]:.2f} '
              f'{(r["mean_p"] or 0):.2f} {(r["clf_band"] or "-"):8s} '
              f'{r["v8_only_verdict"]:7s} {r["cascade_verdict"]:7s} {r["gt"]}')

    # ---- Per-source v8-only vs cascade -----------------------------------
    print('\n=== aggregate: v8-only vs production cascade ===')
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    for src in sorted(by_src):
        rs = by_src[src]
        gt = GT.get(src, 'unknown')
        v8_tumor = sum(1 for r in rs if r['v8_only_verdict'] == 'TUMOR')
        cas_tumor = sum(1 for r in rs if r['cascade_verdict'] == 'TUMOR')
        n = len(rs)
        if gt == 'no_tumor':
            v8_fp = v8_tumor / n
            cas_fp = cas_tumor / n
            print(f'  {src:46s} GT=neg n={n} '
                  f'v8_FP={v8_fp:.0%} -> cascade_FP={cas_fp:.0%}')
        elif gt == 'tumor':
            v8_re = v8_tumor / n
            cas_re = cas_tumor / n
            print(f'  {src:46s} GT=pos n={n} '
                  f'v8_recall={v8_re:.0%} -> cascade_recall={cas_re:.0%}')
        else:
            print(f'  {src:46s} GT=? n={n} v8_TUMOR={v8_tumor} cascade_TUMOR={cas_tumor}')

    # ---- Per-modality breakdown (UniData only — labeled subset) ---------
    print('\n=== per-modality, UniData tumor cohort (GT=tumor) ===')
    mod_rs = {}
    for r in rows:
        if r['source'] != 'tumor_proprietary_multimodal_unidata':
            continue
        mod_rs.setdefault(r['modality'], []).append(r)
    print(f'{"modality":18s}  n   v8_recall  cascade_recall  mean(p_clf)')
    for mod, rs in sorted(mod_rs.items()):
        v8 = sum(1 for r in rs if r['v8_only_verdict'] == 'TUMOR') / len(rs)
        cas = sum(1 for r in rs if r['cascade_verdict'] == 'TUMOR') / len(rs)
        avg_p = np.mean([r['mean_p'] or 0 for r in rs])
        print(f'  {mod:16s}  {len(rs):2d}    {v8:6.0%}        {cas:6.0%}          {avg_p:.3f}')

    # ---- Threshold sweep (reuses cached prob_maps -> ~free) -------------
    print('\n=== v8 threshold sweep on cached prob maps ===')
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    print(f'{"source":46s} GT       ' + '  '.join(f't={t:.2f}' for t in thresholds))
    for src in sorted(by_src):
        rs = by_src[src]
        gt = GT.get(src, 'unknown')
        cells = []
        for t in thresholds:
            tumor_n = sum(1 for r in rs if int((r['prob_map'] >= t).sum()) >= MIN_TUMOR_AREA)
            frac = tumor_n / len(rs)
            cells.append(f'{frac:.0%}'.rjust(6))
        print(f'  {src:44s} {gt:8s}  ' + '  '.join(cells))
    print(f'  (cells = fraction of samples called TUMOR at that threshold; '
          f'for GT=no_tumor that IS the FP rate; for GT=tumor it IS the recall)')

    # ---- Per-modality threshold sweep on UniData -------------------------
    print('\n=== per-modality threshold sweep on UniData (GT=tumor -> recall) ===')
    print(f'{"modality":18s}  n  ' + '  '.join(f't={t:.2f}' for t in thresholds))
    for mod, rs in sorted(mod_rs.items()):
        cells = []
        for t in thresholds:
            tumor_n = sum(1 for r in rs if int((r['prob_map'] >= t).sum()) >= MIN_TUMOR_AREA)
            cells.append(f'{tumor_n / len(rs):.0%}'.rjust(6))
        print(f'  {mod:16s}  {len(rs):2d}  ' + '  '.join(cells))

    # Persist results (without prob_map to keep csv small)
    out_csv = SAMPLES_DIR / 'eval_cascade_results.csv'
    fields = [k for k in rows[0] if k != 'prob_map']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != 'prob_map'})
    print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
