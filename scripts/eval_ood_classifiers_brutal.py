"""Brutal per-classifier OOD audit.

Runs every classifier (cnn, transfer, vit) AND v8 segmentation on every
OOD sample. Reports per-classifier accuracy WITHOUT cascade smoothing.
This is the unfiltered view the dashboard's classifier comparison panel
displays — i.e. what the user actually sees when something looks wrong.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_ood_cascade import (
    SEG_ONNX, CLF_ONNX, MIN_TUMOR_AREA,
    _sess, _preprocess_seg, seg_tta, classify_all, modality_of, GT,
)

SAMPLES_DIR = ROOT / 'samples' / 'ood'


def main():
    seg = _sess(SEG_ONNX)
    clfs = {n: _sess(p) for n, p in CLF_ONNX.items()}
    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples across {len(GT)} sources')
    print(f'[init] each classifier called individually, no consensus, no cascade.\n')

    rows = []
    t0 = time.perf_counter()
    for p in samples:
        img = Image.open(p)
        gt = GT.get(p.parent.name, 'unknown')
        probs = classify_all(clfs, img)
        prob_map = seg_tta(seg, _preprocess_seg(img))
        rows.append({
            'source': p.parent.name,
            'file': p.name,
            'gt': gt,
            'p_cnn': probs['cnn'],
            'p_transfer': probs['transfer'],
            'p_vit': probs['vit'],
            'v8_pmax': float(prob_map.max()),
            'v8_area_020': int((prob_map >= 0.20).sum()),
            'v8_area_030': int((prob_map >= 0.30).sum()),
        })
    print(f'[done] {len(rows)} samples in {time.perf_counter()-t0:.0f}s\n')

    # =================== per-classifier brutal scorecard ===================
    print('='*78)
    print('PER-CLASSIFIER ACCURACY ON OOD (no cascade, no consensus, no overrides)')
    print('='*78)
    for clf in ('cnn', 'transfer', 'vit'):
        pkey = f'p_{clf}'
        # Standard 0.5 threshold for binary classification.
        TP = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]>=0.5)
        FN = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]<0.5)
        FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]>=0.5)
        TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]<0.5)
        recall = TP/(TP+FN) if TP+FN else 0
        fpr = FP/(FP+TN) if FP+TN else 0
        acc = (TP+TN)/len(rows)
        print(f'\n  {clf.upper():12s}  TP={TP:2d}  FN={FN:2d}  FP={FP:2d}  TN={TN:2d}  '
              f'recall={recall:.0%}  FPR={fpr:.0%}  accuracy={acc:.0%}')
        # show worst FN cases (real tumors confidently called no_tumor)
        confidently_wrong = sorted(
            [r for r in rows if r['gt']=='tumor' and r[pkey]<0.2],
            key=lambda r: r[pkey])[:5]
        if confidently_wrong:
            print(f'    {len(confidently_wrong)} cases this classifier said p<0.20 on a real tumor:')
            for r in confidently_wrong:
                print(f'      p={r[pkey]:.2f}  {r["source"][:35]:35s}  {r["file"][:40]}')

    # =================== per-source recall ===============================
    print('\n' + '='*78)
    print('PER-SOURCE: how often each classifier sees the tumor (recall on GT=tumor)')
    print('='*78)
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    print(f'\n{"source":48s} GT     n   {"cnn":>5s} {"trans":>5s} {"vit":>5s}  {"v8seg":>6s}')
    for src in sorted(by_src):
        rs = by_src[src]
        gt = rs[0]['gt']
        n = len(rs)
        cells = []
        for clf in ('p_cnn', 'p_transfer', 'p_vit'):
            hits = sum(1 for r in rs if r[clf] >= 0.5)
            metric = hits/n
            cells.append(f'{metric:.0%}'.rjust(5))
        v8_hits = sum(1 for r in rs if r['v8_area_020'] >= MIN_TUMOR_AREA)
        v8_metric = v8_hits/n
        kind = 'recall' if gt == 'tumor' else 'FP rate'
        print(f'  {src:46s} {gt[:6]:6s} {n:3d}   {cells[0]} {cells[1]} {cells[2]}  '
              f'{v8_metric:.0%}'.rjust(6) + f'   <- {kind}')

    # =================== consensus vote breakdown ========================
    print('\n' + '='*78)
    print('CLASSIFIER CONSENSUS BREAKDOWN ON GT=tumor (the cases that matter most)')
    print('='*78)
    tum = [r for r in rows if r['gt'] == 'tumor']
    print(f'\n  Total OOD tumor samples: {len(tum)}')
    n_all_no = sum(1 for r in tum if r['p_cnn']<0.5 and r['p_transfer']<0.5 and r['p_vit']<0.5)
    n_all_yes = sum(1 for r in tum if r['p_cnn']>=0.5 and r['p_transfer']>=0.5 and r['p_vit']>=0.5)
    n_split = len(tum) - n_all_no - n_all_yes
    print(f'    ALL 3 say "no_tumor":     {n_all_no:3d} / {len(tum)}  ({n_all_no/len(tum):.0%})  <- catastrophic miss')
    print(f'    ALL 3 say "tumor":         {n_all_yes:3d} / {len(tum)}  ({n_all_yes/len(tum):.0%})  <- clean detect')
    print(f'    SPLIT (some yes, some no): {n_split:3d} / {len(tum)}  ({n_split/len(tum):.0%})  <- needs review')

    print('\n  Cases where ALL 3 classifiers confidently miss the tumor:')
    all_miss = sorted([r for r in tum if r['p_cnn']<0.5 and r['p_transfer']<0.5 and r['p_vit']<0.5],
                       key=lambda r: max(r['p_cnn'], r['p_transfer'], r['p_vit']))
    for r in all_miss[:10]:
        print(f'    cnn={r["p_cnn"]:.2f}  trans={r["p_transfer"]:.2f}  vit={r["p_vit"]:.2f}  '
              f'v8_pmax={r["v8_pmax"]:.2f}  v8_area={r["v8_area_020"]:5d}  {r["file"][:50]}')

    # =================== v8-rescues-classifiers ==========================
    print('\n' + '='*78)
    print('THE v8 RESCUE TEST: when ALL 3 classifiers miss, does v8 still find the tumor?')
    print('='*78)
    rescued = sum(1 for r in all_miss
                   if r['v8_area_020'] >= MIN_TUMOR_AREA and r['v8_pmax'] >= 0.70)
    print(f'\n  All-classifier-miss cases: {len(all_miss)}')
    print(f'  Of those, v8 still produces a STRONG positive (area>=50 AND pmax>=0.70):  '
          f'{rescued} / {len(all_miss)} ({rescued/max(1,len(all_miss)):.0%})')
    print(f'  These are the cases the v8_strong override rule (shipped today) catches.')


if __name__ == '__main__':
    main()
