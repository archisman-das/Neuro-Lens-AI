"""Threshold sweep on the v8-retrained classifiers.

Hypothesis: dataset_v8 is 78% positive class, so binary cross-entropy
without pos_weight pushed the new classifiers toward predicting tumor.
The default 0.5 threshold inherits that bias. A higher threshold should
recover specificity without sacrificing too much recall.

Reuses scripts/eval_ood_classifiers_v8_retrained.py's per-image probs;
just sweeps the decision threshold.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_ood_cascade import _sess, _preprocess_clf, GT
from scripts.eval_ood_classifiers_v8_retrained import NEW_CLF_ONNX, NORMALIZE_IMAGENET

SAMPLES_DIR = ROOT / 'samples' / 'ood'


def classify_all(clfs, img):
    out = {}
    for name, sess in clfs.items():
        chw = _preprocess_clf(img, NORMALIZE_IMAGENET[name])
        logit = float(sess.run(None, {sess.get_inputs()[0].name: chw[None]})[0].reshape(-1)[0])
        out[name] = 1.0 / (1.0 + np.exp(-logit))
    return out


def main():
    clfs = {n: _sess(p) for n, p in NEW_CLF_ONNX.items()}
    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                      and p.parent.name in GT)
    rows = []
    t0 = time.perf_counter()
    for p in samples:
        img = Image.open(p)
        gt = GT[p.parent.name]
        probs = classify_all(clfs, img)
        rows.append({'gt': gt, **{f'p_{k}': v for k, v in probs.items()}})
    print(f'[done] {len(rows)} samples in {time.perf_counter()-t0:.0f}s\n')

    print('='*78)
    print('THRESHOLD SWEEP per classifier (recall / FPR)')
    print('='*78)
    THR = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    for clf in ('cnn', 'transfer', 'vit'):
        pkey = f'p_{clf}'
        print(f'\n  {clf.upper()}:')
        print(f'  {"thr":>5s}  {"recall":>7s}  {"FPR":>7s}  {"acc":>6s}  {"F1":>6s}')
        for t in THR:
            TP = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]>=t)
            FN = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]<t)
            FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]>=t)
            TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]<t)
            recall = TP/(TP+FN) if TP+FN else 0
            fpr = FP/(FP+TN) if FP+TN else 0
            acc = (TP+TN)/len(rows)
            f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
            print(f'  {t:>5.2f}  {recall:>7.0%}  {fpr:>7.0%}  {acc:>6.0%}  {f1:>6.2f}')

    # Find optimal threshold per classifier (max F1)
    print('\n' + '='*78)
    print('SUGGESTED OPERATING POINTS (max F1 from sweep)')
    print('='*78)
    for clf in ('cnn', 'transfer', 'vit'):
        pkey = f'p_{clf}'
        best_t, best_f1 = None, -1
        for t in np.arange(0.10, 0.95, 0.02):
            TP = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]>=t)
            FN = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]<t)
            FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]>=t)
            f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
            if f1 > best_f1:
                best_f1 = f1; best_t = t
        # Report at the best
        TP = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]>=best_t)
        FN = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]<best_t)
        FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]>=best_t)
        TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]<best_t)
        recall = TP/(TP+FN); fpr = FP/(FP+TN); acc = (TP+TN)/len(rows)
        print(f'  {clf:10s}  t={best_t:.2f}  recall={recall:.0%}  FPR={fpr:.0%}  '
              f'acc={acc:.0%}  F1={best_f1:.2f}')


if __name__ == '__main__':
    main()
