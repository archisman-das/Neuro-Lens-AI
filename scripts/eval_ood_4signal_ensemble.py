"""4-signal ensemble: can {v9c, v8, symmetry, ANDi.max} hit recall>=95% and FPR<=10%?

Loads merged signals from prior evals:
  - samples/ood/eval_v9c_ensemble_inputs.csv  (v9c_p95, v8_area, sym_p95)
  - samples/ood/eval_v9b_andi_results.csv      (andi max — best ANDi feature
                                                 at AUC 0.726)

Strategy:
  1. Diagnostic — which samples does the v9c high_recall (v9c OR v8) AND sym
     ensemble currently MISS at 94%? Does ANDi.max separate them?
  2. Full sweep over 4-signal rules + thresholds. Report any combination
     hitting recall>=0.95 AND FPR<=0.10.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    v9c_csv = ROOT / 'samples' / 'ood' / 'eval_v9c_ensemble_inputs.csv'
    andi_csv = ROOT / 'samples' / 'ood' / 'eval_v9b_andi_results.csv'

    base = list(csv.DictReader(v9c_csv.open(encoding='utf-8')))
    andi = list(csv.DictReader(andi_csv.open(encoding='utf-8')))
    andi_by_key = {(r['source'], r['file']): r for r in andi}

    rows = []
    for r in base:
        k = (r['source'], r['file'])
        a = andi_by_key.get(k)
        if a is None:
            continue
        rows.append({
            'source': r['source'], 'file': r['file'], 'gt': r['gt'],
            'v9c_p95': float(r['v9c_p95']),
            'v8_area': int(r['v8_area']),
            'sym_p95': float(r['sym_p95']),
            'andi_max': float(a['max']),
            'andi_p95': float(a['p95']),
            'andi_p99': float(a['p99']),
        })
    print(f'[merged] {len(rows)} samples '
          f'({sum(1 for r in rows if r["gt"]=="tumor")} tumor / '
          f'{sum(1 for r in rows if r["gt"]=="no_tumor")} healthy)')

    # ============ Diagnostic: what does v9c high_recall MISS? ============
    print('\n[diagnostic] v9c high_recall = (v9c>0.660 OR v8>=4999) AND sym>53')
    fn = []
    for r in rows:
        v9c = r['v9c_p95'] > 0.660
        v8 = r['v8_area'] >= 4999
        sym = r['sym_p95'] > 53.0
        fires = (v9c or v8) and sym
        if r['gt'] == 'tumor' and not fires:
            fn.append(r)
    print(f'  FN count = {len(fn)} (of 36 tumor cases)')
    for r in fn:
        print(f'    {r["source"]:50s} {r["file"]:30s}  '
              f'v9c={r["v9c_p95"]:.3f} v8={r["v8_area"]:>6d} sym={r["sym_p95"]:>5.1f}  '
              f'andi_max={r["andi_max"]:.2e} andi_p95={r["andi_p95"]:.2e}')

    # Is ANDi.max higher than the healthy cohort's max for these FN?
    healthy_max = [r['andi_max'] for r in rows if r['gt'] == 'no_tumor']
    h_p50, h_p75, h_p90 = (float(np.percentile(healthy_max, p)) for p in (50, 75, 90))
    print(f'  healthy ANDi.max: p50={h_p50:.2e} p75={h_p75:.2e} p90={h_p90:.2e}')
    for r in fn:
        marker = 'CATCHES' if r['andi_max'] > h_p90 else 'misses'
        print(f'    {r["file"]:30s}  andi_max={r["andi_max"]:.2e}  ({marker} at healthy p90)')

    # ============ Full 4-signal sweep ============
    print('\n[sweep] full 4-signal grid...')
    v9c_grid = sorted(set(round(r['v9c_p95'], 3) for r in rows))
    v8_grid = [49, 99, 199, 499, 999, 1999, 4999, 9999]
    sym_grid = sorted(set(round(r['sym_p95'], 1) for r in rows if r['sym_p95'] > 0))
    # ANDi max scale is ~1e-5, so build a dense log-grid
    andi_vals = sorted(r['andi_max'] for r in rows if r['andi_max'] > 0)
    andi_grid = [andi_vals[int(len(andi_vals)*q)] for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)]

    rules = {
        '(v9c OR v8 OR andi) AND sym':     lambda c, v, s, a: (c or v or a) and s,
        '(v9c OR v8) AND (sym OR andi)':   lambda c, v, s, a: (c or v) and (s or a),
        '(v9c OR andi) AND v8':            lambda c, v, s, a: (c or a) and v,
        '(v9c OR andi) AND sym':           lambda c, v, s, a: (c or a) and s,
        '(v9c AND sym) OR (v8 AND andi)':  lambda c, v, s, a: (c and s) or (v and a),
        '(v9c AND sym) OR (andi AND sym)': lambda c, v, s, a: (c and s) or (a and s),
        '2-of-4':                          lambda c, v, s, a: (int(c)+int(v)+int(s)+int(a)) >= 2,
        '3-of-4':                          lambda c, v, s, a: (int(c)+int(v)+int(s)+int(a)) >= 3,
        'v9c OR (v8 AND sym) OR (andi AND sym)':
            lambda c, v, s, a: c or (v and s) or (a and s),
    }

    def _eval(rule_fn, tc, tv, ts, ta):
        TP = FN = FP = TN = 0
        for r in rows:
            fires = rule_fn(r['v9c_p95'] > tc, r['v8_area'] >= tv,
                             r['sym_p95'] > ts, r['andi_max'] > ta)
            if r['gt'] == 'tumor': TP += fires; FN += (not fires)
            else: FP += fires; TN += (not fires)
        re = TP/(TP+FN) if TP+FN else 0
        fp = FP/(FP+TN) if FP+TN else 0
        pr = TP/(TP+FP) if TP+FP else 0
        acc = (TP+TN)/len(rows) if rows else 0
        f1 = 2*pr*re/(pr+re) if pr+re else 0
        return re, fp, acc, f1

    hits = []
    pareto = []
    n_total = len(v9c_grid)*len(v8_grid)*len(sym_grid)*len(andi_grid)*len(rules)
    print(f'  sweeping {n_total:,} combinations...')
    for tc, tv, ts, ta, (name, rule) in product(v9c_grid, v8_grid, sym_grid, andi_grid, rules.items()):
        re, fp, acc, f1 = _eval(rule, tc, tv, ts, ta)
        pareto.append((re, fp, name, tc, tv, ts, ta, f1, acc))
        if re >= 0.95 and fp <= 0.10:
            hits.append((re, fp, name, tc, tv, ts, ta, f1, acc))

    print('\n' + '='*90)
    print(f'TARGET: recall >= 95% AND FPR <= 10%')
    print('='*90)
    if not hits:
        print('  ZERO combinations meet the 95/10 target with 4 signals.')
    else:
        hits.sort(key=lambda x: (-x[7], x[1], -x[0]))
        print(f'  {len(hits)} combinations meet the target!  Top 15 by F1:')
        print(f'  {"rule":40s} {"v9c_t":>6s} {"v8_a":>5s} {"sym_t":>6s} {"andi_t":>10s} '
              f'{"rec":>4s} {"FPR":>4s} {"F1":>5s}')
        for re, fp, name, tc, tv, ts, ta, f1, acc in hits[:15]:
            print(f'  {name:40s} {tc:>6.3f} {tv:>5d} {ts:>6.1f} {ta:>10.3e} '
                  f'{re:>3.0%} {fp:>3.0%} {f1:>4.2f}')

    print('\n' + '='*90)
    print('PARETO FRONTIER: minimum FPR at each recall band')
    print('='*90)
    by_band = defaultdict(list)
    for re, fp, name, tc, tv, ts, ta, f1, acc in pareto:
        band = round(re * 20) / 20
        by_band[band].append((fp, name, tc, tv, ts, ta, f1))
    print(f'  {"recall":>7s}  {"min_FPR":>7s}  {"rule":40s}  {"F1":>5s}')
    for band in sorted(by_band, reverse=True):
        items = sorted(by_band[band])
        fp, name, tc, tv, ts, ta, f1 = items[0]
        print(f'   {band*100:>5.0f}%    {fp*100:>5.1f}%   {name:40s}  {f1:>4.2f}')


if __name__ == '__main__':
    main()
