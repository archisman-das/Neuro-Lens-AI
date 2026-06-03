"""Search for a rule variant that closes the architectural blindspot
revealed by user's failure case 2026-06-03.

Failure case (real OOD scan):
  v9c_p95   = 0.6328    (threshold 0.702 — silent, only 0.07 below)
  v8_area   = 0 px      (threshold 49   — silent, segmenter saw nothing)
  symmetry  = 130       (threshold 83   — FIRED)
  andi_max  = 0.000149  (threshold 1.36e-4 — FIRED)

Current production rule:   (v9c AND sym) OR (v8 AND andi)
                            => both branches dead because the firing
                               pair is (sym AND andi) — the diagonal.

This script tests three families of fixes:
  1. Diagonal-OR — add (sym AND andi) [and optionally (v9c AND andi)]
                    as additional branches.
  2. 2-of-4 voting — any two of four signals fire => tumor.
  3. Soft-OR / weighted — tolerate a near-miss on v9c when sym+andi agree.

For each rule we report the best operating point that:
  (a) keeps recall >= 95% on the 246-sample bench,
  (b) catches the user's failure case,
  (c) minimises FPR.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / 'samples' / 'ood'


def load_rows():
    base = list(csv.DictReader((SAMPLES / 'eval_v9c_ensemble_inputs.csv').open(encoding='utf-8')))
    andi = {(r['source'], r['file']): r
            for r in csv.DictReader((SAMPLES / 'eval_v9b_andi_results.csv').open(encoding='utf-8'))}
    rows = []
    for r in base:
        a = andi.get((r['source'], r['file']))
        if not a:
            continue
        rows.append({
            'gt': r['gt'],
            'v9c': float(r['v9c_p95']),
            'v8': int(r['v8_area']),
            'sym': float(r['sym_p95']),
            'andi': float(a['max']),
        })
    return rows


# The user's actual failure case from their JSON dump
USER_CASE = {'gt': 'tumor', 'v9c': 0.6328, 'v8': 0, 'sym': 130.0, 'andi': 0.000149}


RULES = {
    # Current production
    'baseline: (v9c AND sym) OR (v8 AND andi)':
        lambda c, v, s, a: (c and s) or (v and a),

    # Fix 1: add the missing diagonal pair
    'fix1a: + (sym AND andi)':
        lambda c, v, s, a: (c and s) or (v and a) or (s and a),

    # Fix 1b: add ALL three missing pairs (full pairwise OR)
    'fix1b: + (sym AND andi) + (v9c AND andi)':
        lambda c, v, s, a: (c and s) or (v and a) or (s and a) or (c and a),

    'fix1c: + (sym AND andi) + (v9c AND v8)':
        lambda c, v, s, a: (c and s) or (v and a) or (s and a) or (c and v),

    'fix1d: 2-of-{v9c,sym,andi} OR (v8 AND andi)':
        lambda c, v, s, a: (int(c)+int(s)+int(a) >= 2) or (v and a),

    # Fix 2: any 2 of 4
    'fix2:  2-of-4 voting':
        lambda c, v, s, a: (int(c)+int(v)+int(s)+int(a)) >= 2,

    # Stricter 3-of-4 (for completeness — should under-fire)
    'fix2b: 3-of-4 voting':
        lambda c, v, s, a: (int(c)+int(v)+int(s)+int(a)) >= 3,

    # Soft-OR variants — any single anomaly signal + symmetry confirms
    'fix3a: ((v9c OR andi) AND sym) OR (v8 AND andi)':
        lambda c, v, s, a: ((c or a) and s) or (v and a),

    'fix3b: ((v9c OR andi) AND sym) OR (v8 AND (andi OR sym))':
        lambda c, v, s, a: ((c or a) and s) or (v and (a or s)),

    'fix3c: any-OR (single signal triggers — most aggressive)':
        lambda c, v, s, a: c or v or s or a,
}


def _eval(rule, rows, tc, tv, ts, ta):
    TP = FN = FP = TN = 0
    for r in rows:
        fires = rule(r['v9c'] > tc, r['v8'] >= tv, r['sym'] > ts, r['andi'] > ta)
        if r['gt'] == 'tumor':
            TP += fires; FN += not fires
        else:
            FP += fires; TN += not fires
    re = TP / (TP + FN) if TP + FN else 0
    fp = FP / (FP + TN) if FP + TN else 0
    pr = TP / (TP + FP) if TP + FP else 0
    f1 = 2 * pr * re / (pr + re) if pr + re else 0
    return re, fp, pr, f1


def main():
    rows = load_rows()
    print(f'[loaded] {len(rows)} samples '
          f'({sum(1 for r in rows if r["gt"]=="tumor")} tumor / '
          f'{sum(1 for r in rows if r["gt"]=="no_tumor")} healthy)')
    print(f'[user case] v9c={USER_CASE["v9c"]}  v8={USER_CASE["v8"]}  '
          f'sym={USER_CASE["sym"]}  andi={USER_CASE["andi"]}')

    # Threshold grids — keep the same as production sweep
    v9c_g = sorted(set(round(r['v9c'], 3) for r in rows))
    v8_g = [49, 99, 199, 499, 999, 1999, 4999, 9999]
    sym_g = sorted(set(round(r['sym'], 1) for r in rows if r['sym'] > 0))
    andi_vals = sorted(r['andi'] for r in rows if r['andi'] > 0)
    andi_g = [andi_vals[int(len(andi_vals) * q)]
              for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)]

    n_grid = len(v9c_g) * len(v8_g) * len(sym_g) * len(andi_g)
    print(f'[grid] {n_grid:,} threshold combos per rule\n')

    print('=' * 110)
    print(f'  {"rule":58s} {"target":12s} {"rec":>4s} {"FPR":>5s} {"prec":>5s} {"F1":>5s}  catches?')
    print('=' * 110)

    overall_best = None
    for rname, rule in RULES.items():
        # For each rule, find the best config that meets recall>=95% and
        # also catches the user case.
        best_with_user = None
        best_overall = None
        for tc, tv, ts, ta in product(v9c_g, v8_g, sym_g, andi_g):
            re, fp, pr, f1 = _eval(rule, rows, tc, tv, ts, ta)

            # "Best overall" — best F1 with recall >= 95
            if re >= 0.95:
                key_overall = (fp, -f1)
                if best_overall is None or key_overall < best_overall[0]:
                    best_overall = (key_overall, re, fp, pr, f1, tc, tv, ts, ta)

            # Does this config catch the user case?
            uc = USER_CASE
            catches = rule(uc['v9c'] > tc, uc['v8'] >= tv, uc['sym'] > ts, uc['andi'] > ta)
            if catches and re >= 0.95:
                key_user = (fp, -f1)
                if best_with_user is None or key_user < best_with_user[0]:
                    best_with_user = (key_user, re, fp, pr, f1, tc, tv, ts, ta)

        # Display
        if best_overall:
            _, re, fp, pr, f1, tc, tv, ts, ta = best_overall
            catches_str = 'no'
            if best_with_user:
                catches_str = 'YES (same/diff config)'
            print(f'  {rname:58s} {"re>=0.95":12s} '
                  f'{re:>3.0%} {fp:>4.0%} {pr:>4.0%} {f1:>5.3f}  {catches_str}')
            if best_with_user and best_with_user is not best_overall:
                _, re, fp, pr, f1, tc, tv, ts, ta = best_with_user
                print(f'  {"":58s} {"+catch":12s} '
                      f'{re:>3.0%} {fp:>4.0%} {pr:>4.0%} {f1:>5.3f}  '
                      f'tc={tc:.3f} tv={tv} ts={ts} ta={ta:.2e}')

        # Track overall best across rules — F1-maximising with user-case catch
        if best_with_user:
            entry = (best_with_user[4], best_with_user[2], rname, best_with_user[5:])
            if overall_best is None or entry > overall_best:
                overall_best = entry

    print()
    print('=' * 110)
    print('OPTIMAL RULE (highest F1 while catching the user failure case)')
    print('=' * 110)
    if overall_best is None:
        print('  no rule catches the user case at recall >= 95% — relax constraint')
    else:
        f1_opt, fp_opt, rname, (tc, tv, ts, ta) = overall_best
        re_opt = None
        for r in [overall_best]:
            re_opt = r
        # Re-eval to get re/pr cleanly
        rule = RULES[rname]
        re, fp, pr, f1 = _eval(rule, rows, tc, tv, ts, ta)
        print(f'  rule:        {rname}')
        print(f'  thresholds:  v9c>{tc}  v8>={tv}  sym>{ts}  andi>{ta:.3e}')
        print(f'  measured:    recall={re:.0%}  FPR={fp:.0%}  prec={pr:.0%}  F1={f1:.3f}')
        print(f'  catches the user failure case: YES')
        # Sanity-check: how much does this differ from baseline?
        base_rule = RULES['baseline: (v9c AND sym) OR (v8 AND andi)']
        re_b, fp_b, pr_b, f1_b = _eval(base_rule, rows, 0.702, 49, 83.0, 1.36e-4)
        print(f'\n  vs baseline (current production): '
              f'recall={re_b:.0%} FPR={fp_b:.0%} F1={f1_b:.3f}')
        print(f'  delta:  recall {re-re_b:+.0%}   FPR {fp-fp_b:+.0%}   F1 {f1-f1_b:+.3f}')


if __name__ == '__main__':
    main()
