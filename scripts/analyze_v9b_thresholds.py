"""Diagnose v9b's 100% FPR: per-source p95 distribution + threshold sweep.

The conformal q=0.308 from training-time calibration flagged 12/12 OOD
healthy as anomalies. That doesn't mean the JEPA model is bad — it likely
means the calibration mix (Kaggle 4-class skulls + OpenNeuro skull-
stripped + BraTS no-tumor) has a long tail that the OOD OpenNeuro samples
happen to sit in. This script reads eval_v9b_jepa_results.csv and:

  1. Histograms the per-image p95 by source (tumor vs healthy).
  2. Computes per-image AUC for tumor-vs-healthy separation.
  3. Sweeps thresholds to find the Pareto frontier.
  4. Suggests an operational threshold that keeps recall high while
     dropping FPR to a usable level.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'samples' / 'ood' / 'eval_v9b_jepa_results.csv'


def main():
    rows = list(csv.DictReader(CSV.open(encoding='utf-8')))
    for r in rows:
        for k in ('v9b_p95', 'v9b_max', 'v9b_mean'):
            r[k] = float(r[k])
        r['v9b_ano_area'] = int(r['v9b_ano_area'])

    print('='*82)
    print('per-source p95 distribution (the score that conformal threshold acts on)')
    print('='*82)
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    print(f'\n{"source":48s} GT     n   min      p25     median   p75    max')
    for src in sorted(by_src):
        rs = by_src[src]; gt = rs[0]['gt']
        s = np.array([r['v9b_p95'] for r in rs])
        print(f'  {src:46s} {gt[:6]:6s} {len(rs):3d}   '
              f'{s.min():.3f}   {np.percentile(s,25):.3f}   {np.median(s):.3f}   '
              f'{np.percentile(s,75):.3f}   {s.max():.3f}')

    # ============= AUC for tumor-vs-healthy on p95 =============
    pos = [r['v9b_p95'] for r in rows if r['gt'] == 'tumor']
    neg = [r['v9b_p95'] for r in rows if r['gt'] == 'no_tumor']
    wins = ties = total = 0
    for sp in pos:
        for sn in neg:
            if sp > sn: wins += 1
            elif sp == sn: ties += 1
            total += 1
    auc = (wins + 0.5*ties) / total
    print(f'\n  AUC(p95, tumor vs healthy) on OOD = {auc:.4f}')
    print(f'  perfect separation would be 1.000; random = 0.500')

    # ============= threshold sweep =============
    print('\n' + '='*82)
    print('THRESHOLD SWEEP on v9b p95 score')
    print('='*82)
    print(f'  calibration time q = 0.308 (catches 100% recall but 100% FPR on this OOD set)')
    print()
    thresholds = sorted(set(
        list(np.arange(0.30, 0.60, 0.02))
        + list(np.arange(0.60, 1.20, 0.05))
    ))
    print(f'  {"thr":>6s}   recall    FPR     accuracy  F1     missed_tumor_files')
    pareto_pts = []
    for t in thresholds:
        TP = sum(1 for r in rows if r['gt']=='tumor' and r['v9b_p95'] > t)
        FN = sum(1 for r in rows if r['gt']=='tumor' and r['v9b_p95'] <= t)
        FP = sum(1 for r in rows if r['gt']=='no_tumor' and r['v9b_p95'] > t)
        TN = sum(1 for r in rows if r['gt']=='no_tumor' and r['v9b_p95'] <= t)
        recall = TP/(TP+FN) if TP+FN else 0
        fpr = FP/(FP+TN) if FP+TN else 0
        acc = (TP+TN)/len(rows)
        f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
        missed = sum(1 for r in rows if r['gt']=='tumor' and r['v9b_p95'] <= t)
        print(f'  {t:>6.2f}   {recall:>5.0%}    {fpr:>5.0%}   {acc:>6.0%}   {f1:.2f}   FN={missed}')
        pareto_pts.append((t, recall, fpr, acc, f1))

    # Suggest operating points
    print('\n' + '='*82)
    print('SUGGESTED OPERATING POINTS')
    print('='*82)
    best_f1 = max(pareto_pts, key=lambda p: p[4])
    print(f'\n  best F1:                t={best_f1[0]:.2f}  recall={best_f1[1]:.0%}  '
          f'FPR={best_f1[2]:.0%}  acc={best_f1[3]:.0%}  F1={best_f1[4]:.2f}')
    high_recall = max((p for p in pareto_pts if p[1] >= 0.95), key=lambda p: -p[2], default=None)
    if high_recall:
        print(f'  highest recall >= 95%:  t={high_recall[0]:.2f}  recall={high_recall[1]:.0%}  '
              f'FPR={high_recall[2]:.0%}  acc={high_recall[3]:.0%}  F1={high_recall[4]:.2f}')
    low_fpr = max((p for p in pareto_pts if p[2] <= 0.25), key=lambda p: p[1], default=None)
    if low_fpr:
        print(f'  highest recall @ FPR <=25%: t={low_fpr[0]:.2f}  recall={low_fpr[1]:.0%}  '
              f'FPR={low_fpr[2]:.0%}  acc={low_fpr[3]:.0%}  F1={low_fpr[4]:.2f}')


if __name__ == '__main__':
    main()
