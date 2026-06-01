"""OOD audit on v8-balanced classifiers (OpenNeuro-augmented + pos_weight).

Three-way comparison:
  - OLD:      real_eval_current/        (Kaggle 4-class only)
  - v8-RAW:   real_eval_v8_retrained/   (dataset_v8 as-is, 78% positive)
  - v8-BAL:   real_eval_v8_balanced/    (+ OpenNeuro healthy, pos_weight=0.49)

Hypothesis: v8-balanced fixes the 100% OOD-healthy FPR of v8-raw while
keeping the recall recovery from broader training distribution.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_ood_cascade import SEG_ONNX, MIN_TUMOR_AREA, _sess, _preprocess_seg, _preprocess_clf, seg_tta, GT

CLF_SETS = {
    'OLD':    {'cnn': ROOT/'real_eval_current'/'cnn'/'best_weights.onnx',
                'transfer': ROOT/'real_eval_current'/'transfer'/'best_weights.onnx',
                'vit': ROOT/'real_eval_current'/'vit'/'best_weights.onnx'},
    'v8-RAW': {'cnn': ROOT/'real_eval_v8_retrained'/'cnn'/'best_weights.onnx',
                'transfer': ROOT/'real_eval_v8_retrained'/'transfer'/'best_weights.onnx',
                'vit': ROOT/'real_eval_v8_retrained'/'vit'/'best_weights.onnx'},
    'v8-BAL': {'cnn': ROOT/'real_eval_v8_balanced'/'cnn'/'best_weights.onnx',
                'transfer': ROOT/'real_eval_v8_balanced'/'transfer'/'best_weights.onnx',
                'vit': ROOT/'real_eval_v8_balanced'/'vit'/'best_weights.onnx'},
}
NORMALIZE_IMAGENET = {'cnn': False, 'transfer': True, 'vit': True}
SAMPLES_DIR = ROOT / 'samples' / 'ood'


def classify(sess, img, normalise):
    chw = _preprocess_clf(img, normalise)
    logit = float(sess.run(None, {sess.get_inputs()[0].name: chw[None]})[0].reshape(-1)[0])
    return 1.0 / (1.0 + np.exp(-logit))


def stats_at_thr(rows, pkey, thr=0.5):
    TP = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]>=thr)
    FN = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]<thr)
    FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]>=thr)
    TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]<thr)
    recall = TP/(TP+FN) if TP+FN else 0
    fpr = FP/(FP+TN) if FP+TN else 0
    acc = (TP+TN)/(TP+FN+FP+TN) if (TP+FN+FP+TN) else 0
    return TP, FN, FP, TN, recall, fpr, acc


def main():
    # Load all 3 model sets
    sets = {}
    for tag, paths in CLF_SETS.items():
        if not all(p.exists() for p in paths.values()):
            print(f'[skip] {tag}: missing weights')
            continue
        sets[tag] = {n: _sess(p) for n, p in paths.items()}
    print(f'[init] loaded {len(sets)} classifier sets: {list(sets)}')

    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples\n')

    # Compute all probs once
    rows = []
    t0 = time.perf_counter()
    for p in samples:
        img = Image.open(p)
        gt = GT[p.parent.name]
        rec = {'source': p.parent.name, 'file': p.name, 'gt': gt}
        for tag, clfs in sets.items():
            for n, sess in clfs.items():
                rec[f'{tag}__{n}'] = classify(sess, img, NORMALIZE_IMAGENET[n])
        rows.append(rec)
    print(f'[done] {len(rows)} samples in {time.perf_counter()-t0:.0f}s\n')

    # ==================== summary table ===========================
    print('='*78)
    print('PER-CLASSIFIER SCORECARD (OLD vs v8-RAW vs v8-BAL)')
    print('='*78)
    for clf in ('cnn', 'transfer', 'vit'):
        print(f'\n  {clf.upper()}:')
        print(f'  {"set":10s}  recall   FPR     acc     F1')
        for tag in ('OLD', 'v8-RAW', 'v8-BAL'):
            if tag not in sets: continue
            pkey = f'{tag}__{clf}'
            TP, FN, FP, TN, re, fpr, acc = stats_at_thr(rows, pkey)
            f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
            print(f'  {tag:10s}   {re:.0%}    {fpr:.0%}    {acc:.0%}    {f1:.2f}')

    # ==================== per-source ===============================
    print('\n' + '='*78)
    print('PER-SOURCE on v8-BAL only (the candidate)')
    print('='*78)
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    print(f'\n{"source":48s} GT     n   {"cnn":>5s} {"trans":>5s} {"vit":>5s}')
    for src in sorted(by_src):
        rs = by_src[src]; gt = rs[0]['gt']; n = len(rs)
        kind = 'recall' if gt == 'tumor' else 'FPR'
        cells = []
        for clf in ('cnn', 'transfer', 'vit'):
            pkey = f'v8-BAL__{clf}'
            hits = sum(1 for r in rs if r[pkey] >= 0.5)
            cells.append(f'{hits/n:.0%}'.rjust(5))
        print(f'  {src:46s} {gt[:6]:6s} {n:3d}   {cells[0]} {cells[1]} {cells[2]}    <- {kind}')

    # ==================== consensus on tumor =======================
    print('\n' + '='*78)
    print('CLASSIFIER CONSENSUS on 36 OOD TUMOR SAMPLES')
    print('='*78)
    tum = [r for r in rows if r['gt'] == 'tumor']
    for tag in ('OLD', 'v8-RAW', 'v8-BAL'):
        if tag not in sets: continue
        n_all_yes = sum(1 for r in tum if all(r[f'{tag}__{c}']>=0.5 for c in ('cnn','transfer','vit')))
        n_all_no = sum(1 for r in tum if all(r[f'{tag}__{c}']<0.5 for c in ('cnn','transfer','vit')))
        n_split = len(tum) - n_all_yes - n_all_no
        print(f'  {tag:10s}  all_tumor={n_all_yes:3d} ({n_all_yes/len(tum):.0%})  '
              f'split={n_split:3d} ({n_split/len(tum):.0%})  '
              f'all_no={n_all_no:3d} ({n_all_no/len(tum):.0%})')

    # ==================== consensus on healthy =====================
    neg = [r for r in rows if r['gt'] == 'no_tumor']
    print(f'\nCLASSIFIER CONSENSUS on {len(neg)} OOD HEALTHY SAMPLES')
    print('-'*78)
    for tag in ('OLD', 'v8-RAW', 'v8-BAL'):
        if tag not in sets: continue
        n_all_yes = sum(1 for r in neg if all(r[f'{tag}__{c}']>=0.5 for c in ('cnn','transfer','vit')))
        n_all_no = sum(1 for r in neg if all(r[f'{tag}__{c}']<0.5 for c in ('cnn','transfer','vit')))
        n_split = len(neg) - n_all_yes - n_all_no
        print(f'  {tag:10s}  all_no={n_all_no:3d} ({n_all_no/len(neg):.0%}, correct)  '
              f'split={n_split:3d} ({n_split/len(neg):.0%})  '
              f'all_yes(FP)={n_all_yes:3d} ({n_all_yes/len(neg):.0%})')


if __name__ == '__main__':
    main()
