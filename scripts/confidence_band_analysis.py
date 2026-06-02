"""Do FPs actually have lower confidence than TPs? If yes, define a band.

Hypothesis: when the cascade returns TUMOR, the per-image confidence
signals (v8 seg_max_prob, area, classifier mean_p) separate true
positives from false positives well enough that we can define a third
verdict: REQUIRES_REVIEW (flag for human radiologist).

Methodology:
  1. Run the view-aware cascade on stratified dataset_v8/test (the same
     400-563 sample subset used in eval_id_regression.py) AND the 48
     OOD samples.
  2. For every prediction labelled TUMOR, capture:
        seg_max  - v8 max probability over the image
        seg_area - v8 tumor area at the view-aware threshold
        clf_mean - classifier ensemble mean probability
        clf_max  - max(p_cnn, p_transfer, p_vit)
        gt       - ground truth
  3. Compute TP vs FP separation: ROC AUC for each signal individually
     and for a simple ensemble.
  4. Sweep a 2-d confidence band (seg_max, clf_mean) and report the band
     that maximises (TP_kept - 0.5 * FP_kept) — pareto-favouring recall.
  5. Print: at chosen band, what fraction of FPs get flagged for review
     vs how many TPs we'd accidentally flag.

Run after eval_id_regression.py + eval_ood_cascade.py have populated the
CSVs, or it will recompute from scratch.
"""
from __future__ import annotations

import csv
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
OOD_DIR = ROOT / 'samples' / 'ood'
PER_SOURCE = 50   # smaller stratified sample so analysis runs faster
SEED = 1234

# Source -> GT for OOD
OOD_GT = {
    'healthy_coronal_T1_openneuro':           'no_tumor',
    'tumor_proprietary_multimodal_unidata':   'tumor',
    'tumor_multi_patient_ultralytics':        'tumor',
    'tumor_binary_navoneel_via_miladfa7':     'tumor',
}


def _gt_from_mask(stem: str) -> str:
    mp = TEST_MASK_DIR / f'{stem}.png'
    if not mp.exists():
        return 'unknown'
    m = np.asarray(Image.open(mp).convert('L'))
    return 'tumor' if int((m > 127).sum()) >= MIN_TUMOR_AREA else 'no_tumor'


def _source_of(name: str) -> str:
    if name.startswith('brats_t1c'): return 'brats_t1c'
    if name.startswith('figshare_'): return name.split('_')[0] + '_' + name.split('_')[1]
    if name.startswith('lgg_'): return 'lgg'
    if name.startswith('neg_'): return 'neg_kaggle'
    return name.split('_', 1)[0]


def _gather_samples() -> list[tuple[Path, str, str]]:
    """Return [(path, source, gt), ...] across ID stratified + OOD all."""
    rng = random.Random(SEED)
    out = []
    # ID
    if TEST_IMG_DIR.exists():
        by_src = {}
        for p in TEST_IMG_DIR.glob('*.png'):
            by_src.setdefault(_source_of(p.name), []).append(p)
        for src, pool in sorted(by_src.items()):
            rng.shuffle(pool)
            for p in pool[:PER_SOURCE]:
                out.append((p, f'ID:{src}', _gt_from_mask(p.stem)))
    # OOD
    for p in OOD_DIR.rglob('*'):
        if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        src = p.parent.name
        if src in OOD_GT:
            out.append((p, f'OOD:{src}', OOD_GT[src]))
    return out


def _roc_auc(scores: list[float], labels: list[int]) -> float:
    """Compute ROC AUC where label=1 means TRUE POSITIVE (we want high
    score) and label=0 means FALSE POSITIVE. AUC = P(score_TP > score_FP)."""
    if not scores or not any(labels) or all(labels):
        return float('nan')
    pos = sorted(s for s, l in zip(scores, labels) if l == 1)
    neg = sorted(s for s, l in zip(scores, labels) if l == 0)
    wins, ties, total = 0, 0, 0
    for sp in pos:
        for sn in neg:
            if sp > sn: wins += 1
            elif sp == sn: ties += 1
            total += 1
    return (wins + 0.5 * ties) / total if total else float('nan')


def main():
    samples = _gather_samples()
    print(f'[init] {len(samples)} samples '
          f'(ID stratified={PER_SOURCE}/src + OOD)')

    seg = _sess(SEG_ONNX)
    clfs = {n: _sess(p) for n, p in CLF_ONNX.items()}

    rows = []
    t0 = time.perf_counter()
    last = t0
    for i, (p, src, gt) in enumerate(samples):
        if gt == 'unknown':
            continue
        img = Image.open(p)
        img_rgb = np.asarray(img.convert('RGB'))
        modality = modality_of(p.name)
        view_policy = detect_view(img_rgb,
                                   modality_hint=modality if modality != 'unknown' else None)
        probs = classify_all(clfs, img)
        verdict_c, mean_p, band = consensus(probs)
        prob_map = seg_tta(seg, _preprocess_seg(img))
        seg_max = float(prob_map.max())
        seg_area = int((prob_map >= view_policy.threshold).sum())
        decision, _reason = cascade_decision(
            seg_max_prob=seg_max,
            seg_area_at_view_thresh=seg_area,
            classifier_mean_p=mean_p,
            classifier_band=band,
            view_policy=view_policy,
        )
        # Only analyse predictions that ARE labelled TUMOR by the cascade.
        if decision != 'TUMOR':
            continue
        rows.append({
            'source': src,
            'file': p.name,
            'gt': gt,
            'tp': 1 if gt == 'tumor' else 0,
            'view': view_policy.view,
            'seg_max': seg_max,
            'seg_area': seg_area,
            'clf_mean': mean_p if mean_p is not None else -1.0,
            'clf_max': max(probs.values()),
        })
        if time.perf_counter() - last > 30:
            last = time.perf_counter()
            print(f'  [{i+1}/{len(samples)}] {time.perf_counter()-t0:.0f}s')

    elapsed = time.perf_counter() - t0
    tp_rows = [r for r in rows if r['tp'] == 1]
    fp_rows = [r for r in rows if r['tp'] == 0]
    print(f'\n[done] {len(rows)} TUMOR-labelled predictions in {elapsed:.1f}s')
    print(f'  TP={len(tp_rows)}   FP={len(fp_rows)}')

    if not tp_rows or not fp_rows:
        print('  (insufficient data for separability analysis)')
        return

    # ---- Histograms (binned) ----
    def _binned(values, bins=(0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01)):
        counts = [0] * (len(bins) - 1)
        for v in values:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    counts[i] += 1
                    break
        return counts

    print('\n=== seg_max distribution ===')
    fmt_bins = '[' + ' '.join(f'<{b:.2f}'.rjust(6) for b in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01)) + ']'
    print(f'  bins:  {fmt_bins}')
    print(f'  TP:    {_binned([r["seg_max"] for r in tp_rows])}')
    print(f'  FP:    {_binned([r["seg_max"] for r in fp_rows])}')

    print('\n=== seg_area distribution (px) ===')
    area_bins = (0, 50, 100, 200, 500, 1000, 5000, 100000)
    print(f'  bins:  {area_bins[1:]}')
    print(f'  TP:    {_binned([r["seg_area"] for r in tp_rows], area_bins)}')
    print(f'  FP:    {_binned([r["seg_area"] for r in fp_rows], area_bins)}')

    print('\n=== clf_mean (3-classifier ensemble mean) ===')
    print(f'  bins:  {fmt_bins}')
    print(f'  TP:    {_binned([r["clf_mean"] for r in tp_rows])}')
    print(f'  FP:    {_binned([r["clf_mean"] for r in fp_rows])}')

    # ---- ROC AUC per signal ----
    print('\n=== TP-vs-FP separability (ROC AUC, 1.0 = perfect) ===')
    for sig in ('seg_max', 'seg_area', 'clf_mean', 'clf_max'):
        scores = [r[sig] for r in rows]
        labels = [r['tp'] for r in rows]
        auc = _roc_auc(scores, labels)
        print(f'  {sig:10s}  AUC = {auc:.3f}')

    # Composite score: 0.5 * seg_max + 0.5 * clf_mean
    composite = [0.5 * r['seg_max'] + 0.5 * max(r['clf_mean'], 0) for r in rows]
    auc_comp = _roc_auc(composite, [r['tp'] for r in rows])
    print(f'  seg_max+clf_mean (avg)  AUC = {auc_comp:.3f}')

    # ---- Sweep abstain bands ----
    # A "REQUIRES_REVIEW" predicate: (seg_max < A) OR (clf_mean < B AND seg_area < C)
    # We want it to capture many FPs but few TPs.
    print('\n=== sweep candidate REQUIRES_REVIEW rules ===')
    print('  REVIEW when: (seg_max < A_thr) OR (clf_mean < 0.30 AND seg_area < 200)')
    print(f'  {"A_thr":>6s}  FP_flagged   TP_flagged   net_score')
    best = None
    for A in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        fp_flagged = sum(1 for r in fp_rows
                          if r['seg_max'] < A or (r['clf_mean'] < 0.30 and r['seg_area'] < 200))
        tp_flagged = sum(1 for r in tp_rows
                          if r['seg_max'] < A or (r['clf_mean'] < 0.30 and r['seg_area'] < 200))
        fp_frac = fp_flagged / len(fp_rows)
        tp_frac = tp_flagged / len(tp_rows)
        # Net score: weight FP-flagged 2x because flagging an FP saves a false alarm
        # but flagging a TP just delays a real finding.
        net = 2 * fp_frac - tp_frac
        line = f'  {A:>6.2f}  {fp_flagged:3d}/{len(fp_rows)} ({fp_frac:.0%})    ' \
               f'{tp_flagged:3d}/{len(tp_rows)} ({tp_frac:.0%})    {net:+.2f}'
        if best is None or net > best[0]:
            best = (net, A, fp_flagged, tp_flagged, fp_frac, tp_frac)
        print(line)
    print(f'\n  ===> best A_thr={best[1]:.2f}  net={best[0]:+.2f}  '
          f'(flags {best[2]} FPs / {best[3]} TPs)')

    # ---- Persist for further analysis ----
    out_csv = ROOT / 'samples' / 'ood' / 'confidence_analysis.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
