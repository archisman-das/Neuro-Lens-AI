"""3-way OOD comparison: v8-only vs current cascade vs view-aware cascade.

Reuses scripts/eval_ood_cascade.py for v8 + classifier inference, then
applies src/research/view_router.py to derive a per-image view + threshold
+ classifier-trust decision.

Shows: per-source FP/recall under each policy, and a per-image table so
we can see exactly which images flipped verdicts under the new policy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse: I/O helpers, GT map, constants
from scripts.eval_ood_cascade import (
    SEG_ONNX, CLF_ONNX, NORMALIZE_IMAGENET,
    SAMPLES_DIR, SEG_SIZE, CLF_SIZE, MIN_TUMOR_AREA, IM_MEAN, IM_STD,
    GT, _sess, _preprocess_seg, _preprocess_clf, seg_tta, classify_all,
    consensus, modality_of,
)
from src.research.view_router import detect_view, cascade_decision


def _classifier_only_decision(probs: dict) -> str:
    """Mirror original dashboard cascade decision."""
    verdict, mean_p, band = consensus(probs)
    if verdict == 'no_tumor' and band in ('high', 'moderate'):
        return 'no_tumor'  # suppressed
    if verdict == 'tumor' and band in ('high', 'moderate'):
        return 'TUMOR'
    return 'TUMOR_or_no_tumor_check_seg'


def main():
    seg = _sess(SEG_ONNX)
    clfs = {n: _sess(p) for n, p in CLF_ONNX.items()}
    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    print(f'[init] {len(samples)} OOD samples across {len(GT)} known sources')

    t0 = time.perf_counter()
    rows = []
    for p in samples:
        img = Image.open(p)
        img_rgb = np.asarray(img.convert('RGB'))
        modality = modality_of(p.name)
        policy = detect_view(img_rgb, modality_hint=modality if modality != 'unknown' else None)
        probs = classify_all(clfs, img)
        verdict_c, mean_p, band = consensus(probs)
        prob_map = seg_tta(seg, _preprocess_seg(img))
        seg_max = float(prob_map.max())

        # Three thresholds: fixed 0.20 (v8-only / original cascade) + view-aware.
        area_020 = int((prob_map >= 0.20).sum())
        area_view = int((prob_map >= policy.threshold).sum())

        # Policy A: v8-only at 0.20
        v8_only = 'TUMOR' if area_020 >= MIN_TUMOR_AREA else 'no_tumor'

        # Policy B: current cascade (v8@0.20 + classifier suppression)
        if verdict_c == 'no_tumor' and band in ('high', 'moderate'):
            current_cascade = 'no_tumor'
        elif verdict_c == 'tumor' and band in ('high', 'moderate'):
            current_cascade = 'TUMOR'
        else:
            current_cascade = 'TUMOR' if area_020 >= MIN_TUMOR_AREA else 'no_tumor'

        # Policy C: view-aware cascade
        view_aware, reason = cascade_decision(
            seg_max_prob=seg_max,
            seg_area_at_view_thresh=area_view,
            classifier_mean_p=mean_p,
            classifier_band=band,
            view_policy=policy,
        )

        rows.append({
            'source': p.parent.name,
            'file': p.name,
            'gt': GT.get(p.parent.name, 'unknown'),
            'modality': modality,
            'view': policy.view,
            'view_conf': policy.confidence,
            'thresh_used': policy.threshold,
            'trust_clf': policy.trust_classifier,
            'mean_p': round(mean_p, 3) if mean_p is not None else None,
            'band': band or '-',
            'seg_max': round(seg_max, 3),
            'area_020': area_020,
            'area_view': area_view,
            'v8_only': v8_only,
            'current_cascade': current_cascade,
            'view_aware_cascade': view_aware,
            'reason': reason,
        })
    elapsed = time.perf_counter() - t0

    # ---- Aggregate per source --------------------------------------------
    print('\n=== aggregate per source ===')
    print(f'{"source":48s} GT       n   v8only  current  view-aware')
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    weighted = {'v8_only': 0, 'current_cascade': 0, 'view_aware_cascade': 0}
    weighted_n = 0
    for src in sorted(by_src):
        rs = by_src[src]
        gt = rs[0]['gt']
        n = len(rs)
        a = sum(1 for r in rs if r['v8_only'] == 'TUMOR') / n
        b = sum(1 for r in rs if r['current_cascade'] == 'TUMOR') / n
        c = sum(1 for r in rs if r['view_aware_cascade'] == 'TUMOR') / n
        if gt == 'no_tumor':
            print(f'  {src:46s} neg     {n:3d}   FP={a:.0%}   FP={b:.0%}    FP={c:.0%}')
        else:
            print(f'  {src:46s} pos     {n:3d}   re={a:.0%}   re={b:.0%}    re={c:.0%}')

    # ---- Weighted totals for tumor cohort --------------------------------
    tum_rows = [r for r in rows if r['gt'] == 'tumor']
    neg_rows = [r for r in rows if r['gt'] == 'no_tumor']
    print('\n=== weighted totals ===')
    for label, rs in [('TUMOR (17 OOD instances)', tum_rows),
                       ('HEALTHY (12 OOD subjects)', neg_rows)]:
        if not rs:
            continue
        a = sum(1 for r in rs if r['v8_only'] == 'TUMOR') / len(rs)
        b = sum(1 for r in rs if r['current_cascade'] == 'TUMOR') / len(rs)
        c = sum(1 for r in rs if r['view_aware_cascade'] == 'TUMOR') / len(rs)
        kind = 'recall' if 'TUMOR' in label else 'FP'
        print(f'  {label:32s}: v8_only={a:.0%}  current_cascade={b:.0%}  '
              f'view_aware={c:.0%}  ({kind})')

    # ---- View detection breakdown -----------------------------------------
    print('\n=== view detection (rule-based) ===')
    view_counts = {}
    for r in rows:
        view_counts.setdefault(r['view'], []).append(r)
    for v, rs in sorted(view_counts.items()):
        sources = {r['source'][:30]: 0 for r in rs}
        for r in rs:
            sources[r['source'][:30]] += 1
        s = ', '.join(f'{k}={v}' for k, v in sources.items())
        print(f'  {v:10s} n={len(rs):3d}  ({s})')

    # ---- Per-image diffs (where view-aware flips verdict) ----------------
    print('\n=== verdicts where view-aware DIFFERS from current cascade ===')
    flipped = [r for r in rows if r['view_aware_cascade'] != r['current_cascade']]
    if not flipped:
        print('  (none)')
    else:
        print(f'{"file":52s} {"view":9s} {"thr":>5s} {"GT":>8s}  '
              f'{"current":>8s}  {"view":>8s}  reason')
        for r in flipped:
            print(f'{r["file"][:52]:52s} {r["view"]:9s} '
                  f'{r["thresh_used"]:.2f}  {r["gt"]:>8s}  '
                  f'{r["current_cascade"]:>8s}  {r["view_aware_cascade"]:>8s}  {r["reason"]}')

    print(f'\n[done] {len(rows)} samples in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
