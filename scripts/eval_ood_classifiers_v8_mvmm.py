"""Final OOD audit: v8-mvmm (multi-view multi-modal) classifiers.

4-way comparison:
  - OLD:     real_eval_current/         (Kaggle 4-class only)
  - v8-RAW:  real_eval_v8_retrained/    (dataset_v8 axial-T1c, 78% positive)
  - v8-BAL:  real_eval_v8_balanced/     (+ OpenNeuro healthy, pos_weight)
  - v8-MVMM: real_eval_v8_mvmm/         (+ BraTS sag/cor/T1/T2/FLAIR)

This is the round where the training data actually contains the
acquisition geometries and modalities the OOD set tests on.
Target: ≥70% OOD tumor recall while keeping FPR <30%.
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
    'OLD':     {m: ROOT/'real_eval_current'/m/'best_weights.onnx' for m in ('cnn','transfer','vit')},
    'v8-RAW':  {m: ROOT/'real_eval_v8_retrained'/m/'best_weights.onnx' for m in ('cnn','transfer','vit')},
    'v8-BAL':  {m: ROOT/'real_eval_v8_balanced'/m/'best_weights.onnx' for m in ('cnn','transfer','vit')},
    'v8-MVMM': {m: ROOT/'real_eval_v8_mvmm'/m/'best_weights.onnx' for m in ('cnn','transfer','vit')},
}
NORMALIZE_IMAGENET = {'cnn': False, 'transfer': True, 'vit': True}
SAMPLES_DIR = ROOT / 'samples' / 'ood'


def classify(sess, img, normalise):
    chw = _preprocess_clf(img, normalise)
    logit = float(sess.run(None, {sess.get_inputs()[0].name: chw[None]})[0].reshape(-1)[0])
    return 1.0 / (1.0 + np.exp(-logit))


def stats(rows, pkey, thr=0.5):
    TP = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]>=thr)
    FN = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]<thr)
    FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]>=thr)
    TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]<thr)
    return TP, FN, FP, TN


def main():
    sets = {}
    for tag, paths in CLF_SETS.items():
        if not all(p.exists() for p in paths.values()):
            print(f'[skip] {tag}: missing weights')
            continue
        sets[tag] = {n: _sess(p) for n, p in paths.items()}
    print(f'[init] loaded {len(sets)} classifier sets: {list(sets)}\n')

    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png','.jpg','.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples')

    rows = []
    t0 = time.perf_counter()
    for p in samples:
        img = Image.open(p)
        rec = {'source': p.parent.name, 'file': p.name, 'gt': GT[p.parent.name]}
        for tag, clfs in sets.items():
            for n, sess in clfs.items():
                rec[f'{tag}__{n}'] = classify(sess, img, NORMALIZE_IMAGENET[n])
        rows.append(rec)
    print(f'[done] {time.perf_counter()-t0:.0f}s\n')

    # ============= per-classifier across all 4 sets =================
    print('='*84)
    print('PER-CLASSIFIER OOD SCORECARD')
    print('='*84)
    for clf in ('cnn', 'transfer', 'vit'):
        print(f'\n  {clf.upper()}:')
        print(f'  {"set":10s}  recall   FPR    acc    F1')
        for tag in ('OLD', 'v8-RAW', 'v8-BAL', 'v8-MVMM'):
            if tag not in sets: continue
            TP, FN, FP, TN = stats(rows, f'{tag}__{clf}')
            re = TP/(TP+FN) if TP+FN else 0
            fp = FP/(FP+TN) if FP+TN else 0
            acc = (TP+TN)/(TP+FN+FP+TN)
            f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
            marker = ' <-- new' if tag == 'v8-MVMM' else ''
            print(f'  {tag:10s}  {re:>5.0%}   {fp:>4.0%}   {acc:>4.0%}   {f1:.2f}{marker}')

    # ============= per-source on v8-MVMM (the candidate) ============
    print('\n' + '='*84)
    print('PER-SOURCE on v8-MVMM (the new candidate)')
    print('='*84)
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    print(f'\n{"source":48s} GT     n     cnn  trans   vit')
    for src in sorted(by_src):
        rs = by_src[src]; gt = rs[0]['gt']; n = len(rs)
        kind = 'recall' if gt=='tumor' else 'FPR'
        cells = []
        for c in ('cnn', 'transfer', 'vit'):
            hits = sum(1 for r in rs if r[f'v8-MVMM__{c}'] >= 0.5)
            cells.append(f'{hits/n:.0%}'.rjust(5))
        print(f'  {src:46s} {gt[:6]:6s} {n:3d}    {cells[0]} {cells[1]} {cells[2]}    <- {kind}')

    # ============= tumor consensus across all 4 sets ================
    print('\n' + '='*84)
    print('CONSENSUS on 36 OOD TUMOR SAMPLES — improvement progression')
    print('='*84)
    tum = [r for r in rows if r['gt'] == 'tumor']
    for tag in ('OLD', 'v8-RAW', 'v8-BAL', 'v8-MVMM'):
        if tag not in sets: continue
        all_yes = sum(1 for r in tum if all(r[f'{tag}__{c}']>=0.5 for c in ('cnn','transfer','vit')))
        all_no = sum(1 for r in tum if all(r[f'{tag}__{c}']<0.5 for c in ('cnn','transfer','vit')))
        print(f'  {tag:10s}  all_3_say_tumor={all_yes:3d}/{len(tum)} ({all_yes/len(tum):.0%})   '
              f'all_3_say_no_tumor(catastrophic_miss)={all_no:3d}/{len(tum)} ({all_no/len(tum):.0%})')

    neg = [r for r in rows if r['gt'] == 'no_tumor']
    print(f'\nCONSENSUS on {len(neg)} OOD HEALTHY (OpenNeuro coronal T1)')
    print('-'*84)
    for tag in ('OLD', 'v8-RAW', 'v8-BAL', 'v8-MVMM'):
        if tag not in sets: continue
        all_no = sum(1 for r in neg if all(r[f'{tag}__{c}']<0.5 for c in ('cnn','transfer','vit')))
        all_yes = sum(1 for r in neg if all(r[f'{tag}__{c}']>=0.5 for c in ('cnn','transfer','vit')))
        print(f'  {tag:10s}  all_3_correctly_no_tumor={all_no:3d}/{len(neg)} ({all_no/len(neg):.0%})   '
              f'all_3_wrongly_say_tumor(catastrophic_FP)={all_yes:3d}/{len(neg)} ({all_yes/len(neg):.0%})')


if __name__ == '__main__':
    main()
