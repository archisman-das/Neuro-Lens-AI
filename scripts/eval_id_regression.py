"""In-distribution regression test for the view-aware cascade policy.

Question: does the view-aware policy (src/research/view_router.py) hurt
performance on the dataset_v8 test split — the very distribution v8 was
calibrated for? If yes, we should NOT wire it into the dashboard.

Sample: stratified random ~100 per source from dataset_v8/test, where
masks tell us GT (mask sum > 50px -> tumor, else no_tumor).

Compares three policies side-by-side (same as eval_ood_view_aware.py):
  A) v8-only @ t=0.20
  B) current cascade (v8@0.20 + classifier consensus suppression)
  C) view-aware cascade (view-detect -> per-view threshold + override)

Per-source numbers + an aggregate confusion table. A regression alert
fires if (C) is worse than (B) on aggregate FP or recall by > 3 pp.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_ood_cascade import (
    SEG_ONNX, CLF_ONNX, MIN_TUMOR_AREA,
    _sess, _preprocess_seg, seg_tta, classify_all, consensus, modality_of,
)
from src.research.view_router import detect_view, cascade_decision

TEST_IMG_DIR = ROOT / 'dataset_v8' / 'test' / 'images'
TEST_MASK_DIR = ROOT / 'dataset_v8' / 'test' / 'masks'
PER_SOURCE = 100   # stratified sample size per prefix
SEED = 1234


def _gt_from_mask(stem: str) -> str:
    mp = TEST_MASK_DIR / f'{stem}.png'
    if not mp.exists():
        return 'unknown'
    m = np.asarray(Image.open(mp).convert('L'))
    return 'tumor' if int((m > 127).sum()) >= MIN_TUMOR_AREA else 'no_tumor'


def _source_of(name: str) -> str:
    if name.startswith('brats_t1c'): return 'brats_t1c'
    if name.startswith('figshare_glioma'): return 'figshare_glioma'
    if name.startswith('figshare_meningioma'): return 'figshare_meningioma'
    if name.startswith('figshare_pituitary'): return 'figshare_pituitary'
    if name.startswith('lgg_'): return 'lgg'
    if name.startswith('neg_'): return 'neg_kaggle'
    return name.split('_', 1)[0]


def main():
    rng = random.Random(SEED)
    by_src: dict[str, list[Path]] = {}
    for p in TEST_IMG_DIR.glob('*.png'):
        by_src.setdefault(_source_of(p.name), []).append(p)
    samples: list[Path] = []
    for src in sorted(by_src):
        pool = by_src[src]
        rng.shuffle(pool)
        samples.extend(pool[:PER_SOURCE])
    print(f'[init] stratified sample: {len(samples)} from '
          f'{[(s, min(PER_SOURCE, len(by_src[s]))) for s in sorted(by_src)]}')

    seg = _sess(SEG_ONNX)
    clfs = {n: _sess(p) for n, p in CLF_ONNX.items()}
    rows = []
    t0 = time.perf_counter()
    last_print = t0
    for i, p in enumerate(samples):
        img = Image.open(p)
        gt = _gt_from_mask(p.stem)
        modality = modality_of(p.name)
        view_policy = detect_view(np.asarray(img.convert('RGB')),
                                   modality_hint=modality if modality != 'unknown' else None)
        probs = classify_all(clfs, img)
        verdict_c, mean_p, band = consensus(probs)
        prob_map = seg_tta(seg, _preprocess_seg(img))
        seg_max = float(prob_map.max())
        area_020 = int((prob_map >= 0.20).sum())
        area_view = int((prob_map >= view_policy.threshold).sum())

        # A) v8-only
        v8_only = 'TUMOR' if area_020 >= MIN_TUMOR_AREA else 'no_tumor'
        # B) current cascade
        if verdict_c == 'no_tumor' and band in ('high', 'moderate'):
            current = 'no_tumor'
        elif verdict_c == 'tumor' and band in ('high', 'moderate'):
            current = 'TUMOR'
        else:
            current = 'TUMOR' if area_020 >= MIN_TUMOR_AREA else 'no_tumor'
        # C) view-aware
        view_aware, _reason = cascade_decision(
            seg_max_prob=seg_max,
            seg_area_at_view_thresh=area_view,
            classifier_mean_p=mean_p,
            classifier_band=band,
            view_policy=view_policy,
        )
        rows.append({
            'source': _source_of(p.name), 'file': p.name, 'gt': gt,
            'view': view_policy.view, 'thresh': view_policy.threshold,
            'mean_p': mean_p, 'band': band,
            'v8_only': v8_only, 'current': current, 'view_aware': view_aware,
        })
        # Progress every 30s
        if time.perf_counter() - last_print > 30:
            last_print = time.perf_counter()
            print(f'  [{i+1}/{len(samples)}]  elapsed={time.perf_counter()-t0:.0f}s')

    elapsed = time.perf_counter() - t0
    print(f'[done] {len(rows)} samples in {elapsed:.1f}s ({elapsed/len(rows):.2f}/sample)\n')

    # ---- Per-source per-policy aggregates ----
    def _stats(rs, col):
        gts = [r['gt'] for r in rs]
        preds = [r[col] for r in rs]
        TP = sum(1 for g, p in zip(gts, preds) if g == 'tumor' and p == 'TUMOR')
        FN = sum(1 for g, p in zip(gts, preds) if g == 'tumor' and p == 'no_tumor')
        FP = sum(1 for g, p in zip(gts, preds) if g == 'no_tumor' and p == 'TUMOR')
        TN = sum(1 for g, p in zip(gts, preds) if g == 'no_tumor' and p == 'no_tumor')
        recall = TP / (TP + FN) if (TP + FN) else None
        fpr = FP / (FP + TN) if (FP + TN) else None
        return TP, FN, FP, TN, recall, fpr

    print('=== per-source: recall (sensitivity) / FPR ===')
    print(f'{"source":22s} n   {"v8only":>14s}   {"current":>14s}   {"view_aware":>14s}')
    by = {}
    for r in rows:
        by.setdefault(r['source'], []).append(r)
    for src in sorted(by):
        rs = by[src]
        cells = []
        for col in ('v8_only', 'current', 'view_aware'):
            TP, FN, FP, TN, re, fpr = _stats(rs, col)
            re_s = f'{re:.0%}' if re is not None else ' - '
            fp_s = f'{fpr:.0%}' if fpr is not None else ' - '
            cells.append(f'r={re_s}/f={fp_s}'.rjust(14))
        print(f'  {src:20s} {len(rs):3d}   {cells[0]}   {cells[1]}   {cells[2]}')

    # ---- Aggregate
    print('\n=== ID-aggregate confusion (all sources combined) ===')
    print(f'{"policy":18s}  TP    FN    FP    TN     recall   FPR    accuracy')
    for col in ('v8_only', 'current', 'view_aware'):
        TP, FN, FP, TN, re, fpr = _stats(rows, col)
        acc = (TP + TN) / len(rows)
        print(f'  {col:16s}  {TP:4d}  {FN:4d}  {FP:4d}  {TN:4d}   '
              f'{(re or 0):.1%}   {(fpr or 0):.1%}   {acc:.1%}')

    # ---- Regression alert
    print('\n=== regression check (view_aware vs current) ===')
    _, _, _, _, re_cur, fpr_cur = _stats(rows, 'current')
    _, _, _, _, re_va, fpr_va = _stats(rows, 'view_aware')
    d_re = (re_va or 0) - (re_cur or 0)
    d_fp = (fpr_va or 0) - (fpr_cur or 0)
    print(f'  d_recall = {d_re:+.1%}    (positive = better)')
    print(f'  d_FPR    = {d_fp:+.1%}    (negative = better)')
    if d_re < -0.03:
        print(f'  [REGRESSION] recall dropped by {abs(d_re):.1%} (>3pp threshold)')
    if d_fp > 0.03:
        print(f'  [REGRESSION] FPR rose by {d_fp:.1%} (>3pp threshold)')
    if d_re >= -0.03 and d_fp <= 0.03:
        print(f'  [OK] no significant regression on ID data')


if __name__ == '__main__':
    main()
