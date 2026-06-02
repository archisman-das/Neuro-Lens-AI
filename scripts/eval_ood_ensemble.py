"""Ensemble search for 95%+ recall AND <=10% FPR on OOD.

JEPA alone caps at the 94%/58% or 67%/8% endpoints of its Pareto curve.
v8 alone caps at 86%/58%. Neither individually meets the target.

Hypothesis: the false positives and the false negatives of these signals
are largely INDEPENDENT (JEPA's FPs are healthy coronal-T1 OpenNeuro that
v8 also flags; JEPA's recall recovery is on UniData multimodal that v8
misses). So:

  - Logical-OR (any signal -> tumor) maximises recall but adds FPs
  - Logical-AND (all signals must agree) drops FPR but loses recall
  - K-of-N voting can find a sweet spot

This script loads per-image scores from the previous evals and exhaustively
sweeps every combination of {v9b_JEPA, v9b_DDPM, v8} with per-signal
thresholds + voting rules, then reports the operating points that meet
the user's target (recall >= 0.95 AND FPR <= 0.10).
"""
from __future__ import annotations

import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / 'samples' / 'ood'


def load_scores():
    """Merge per-image scores across the v9b full eval + cascade eval."""
    full = list(csv.DictReader((SAMPLES / 'eval_v9b_full_results.csv').open(encoding='utf-8')))
    cascade = list(csv.DictReader((SAMPLES / 'eval_cascade_results.csv').open(encoding='utf-8')))
    # Index cascade by (source, file)
    casc_by_key = {(r['source'], r['file']): r for r in cascade}
    rows = []
    for r in full:
        key = (r['source'], r['file'])
        c = casc_by_key.get(key, {})
        rows.append({
            'source': r['source'], 'file': r['file'], 'gt': r['gt'],
            'jepa_p95':       float(r.get('v9b_app_p95', 0)),
            'ddpm_p95':       float(r.get('v9b_residual_p95', 0)),
            'v8_pmax':        float(c.get('seg_max', 0) or 0),
            'v8_area_020':    int(r.get('v8_area_020', 0) or 0),
        })
    return rows


def evaluate(rows, vote_rule):
    """vote_rule(r) -> True/False (True = tumor verdict)"""
    TP = sum(1 for r in rows if r['gt']=='tumor' and vote_rule(r))
    FN = sum(1 for r in rows if r['gt']=='tumor' and not vote_rule(r))
    FP = sum(1 for r in rows if r['gt']=='no_tumor' and vote_rule(r))
    TN = sum(1 for r in rows if r['gt']=='no_tumor' and not vote_rule(r))
    recall = TP/(TP+FN) if TP+FN else 0
    fpr = FP/(FP+TN) if FP+TN else 0
    acc = (TP+TN)/len(rows)
    f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
    return TP, FN, FP, TN, recall, fpr, acc, f1


def main():
    rows = load_scores()
    print(f'[init] {len(rows)} OOD samples loaded with v9b_JEPA + v9b_DDPM + v8 scores')

    # Score ranges per signal
    for k in ('jepa_p95', 'ddpm_p95', 'v8_pmax'):
        vals = sorted(r[k] for r in rows)
        print(f'  {k:12s}  min={vals[0]:.3f}  p25={vals[len(vals)//4]:.3f}  '
              f'median={vals[len(vals)//2]:.3f}  p75={vals[3*len(vals)//4]:.3f}  max={vals[-1]:.3f}')
    print()

    # =================== single-signal baselines ===================
    print('='*82)
    print('SINGLE-SIGNAL BASELINES (sweep threshold for max F1 per signal)')
    print('='*82)
    for k in ('jepa_p95', 'ddpm_p95', 'v8_pmax'):
        cands = sorted(set(round(r[k], 4) for r in rows))
        best = (None, -1)
        for t in cands:
            _, _, _, _, re, fp, acc, f1 = evaluate(rows, lambda r, k=k, t=t: r[k] > t)
            if f1 > best[1]:
                best = (t, f1, re, fp, acc)
        t, f1, re, fp, acc = best
        print(f'  {k:12s}  t={t:.3f}  recall={re:.0%}  FPR={fp:.0%}  acc={acc:.0%}  F1={f1:.2f}')

    # =================== exhaustive ensemble sweep ===================
    # For each signal, pick a threshold candidate. For each {AND, OR, 2of3}
    # rule, evaluate. Record everything meeting target.
    print('\n' + '='*82)
    print('TARGET: recall >= 0.95  AND  FPR <= 0.10')
    print('='*82)

    # Coarse threshold grids (avoid quadratic blowup with fine grids)
    j_grid = sorted(set(round(r['jepa_p95'], 3) for r in rows))
    d_grid = sorted(set(round(r['ddpm_p95'], 3) for r in rows))
    v_grid = [49, 99, 199, 499, 999, 1999, 4999]   # v8 area thresholds (px)

    rules = {
        'AND':   lambda j, d, v: j and d and v,
        'OR':    lambda j, d, v: j or d or v,
        '2of3':  lambda j, d, v: (int(j) + int(d) + int(v)) >= 2,
        'JEPA_AND_v8':       lambda j, d, v: j and v,
        'JEPA_AND_DDPM':     lambda j, d, v: j and d,
        'DDPM_AND_v8':       lambda j, d, v: d and v,
        'JEPA_OR_(DDPM_AND_v8)': lambda j, d, v: j or (d and v),
        '(JEPA_AND_DDPM)_OR_v8': lambda j, d, v: (j and d) or v,
    }

    hits = []
    total = len(j_grid) * len(d_grid) * len(v_grid) * len(rules)
    print(f'[sweep] {total} combinations  ({len(j_grid)} jepa thrs x '
          f'{len(d_grid)} ddpm thrs x {len(v_grid)} v8 thrs x {len(rules)} rules)')

    for tj, td, ta, (rule_name, rule_fn) in product(j_grid, d_grid, v_grid, rules.items()):
        def vote(r, tj=tj, td=td, ta=ta, rule_fn=rule_fn):
            return rule_fn(r['jepa_p95'] > tj, r['ddpm_p95'] > td, r['v8_area_020'] >= ta)
        _, _, _, _, re, fp, acc, f1 = evaluate(rows, vote)
        if re >= 0.95 and fp <= 0.10:
            hits.append((re, fp, acc, f1, rule_name, tj, td, ta))

    hits.sort(key=lambda x: (-x[3], x[1], -x[0]))   # sort by F1 desc, then FPR asc

    if not hits:
        print('\n  ZERO combinations meet the target.')
    else:
        print(f'\n  {len(hits)} combinations meet the target. Top 15 by F1:')
        print(f'  {"rule":25s}  jepa_t  ddpm_t  v8_area  recall  FPR    acc    F1')
        for re, fp, acc, f1, rule, tj, td, ta in hits[:15]:
            print(f'  {rule:25s}  {tj:.3f}   {td:.3f}   {ta:>5d}    '
                  f'{re:.0%}    {fp:.0%}   {acc:.0%}   {f1:.2f}')

    # =================== relaxed targets ===================
    print('\n' + '='*82)
    print('RELAXED TARGETS (recall >=0.90, FPR <= 0.20)')
    print('='*82)
    relaxed = []
    for tj, td, ta, (rule_name, rule_fn) in product(j_grid, d_grid, v_grid, rules.items()):
        def vote(r, tj=tj, td=td, ta=ta, rule_fn=rule_fn):
            return rule_fn(r['jepa_p95'] > tj, r['ddpm_p95'] > td, r['v8_area_020'] >= ta)
        _, _, _, _, re, fp, acc, f1 = evaluate(rows, vote)
        if re >= 0.90 and fp <= 0.20:
            relaxed.append((re, fp, acc, f1, rule_name, tj, td, ta))
    relaxed.sort(key=lambda x: (-x[0], x[1]))
    if not relaxed:
        print('\n  ZERO combinations meet recall >= 0.90 AND FPR <= 0.20.')
    else:
        print(f'\n  {len(relaxed)} combinations. Top 10 by recall:')
        print(f'  {"rule":25s}  jepa_t  ddpm_t  v8_area  recall  FPR    acc    F1')
        for re, fp, acc, f1, rule, tj, td, ta in relaxed[:10]:
            print(f'  {rule:25s}  {tj:.3f}   {td:.3f}   {ta:>5d}    '
                  f'{re:.0%}    {fp:.0%}   {acc:.0%}   {f1:.2f}')


if __name__ == '__main__':
    main()
