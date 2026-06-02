"""Full v9b eval including the new symmetry-based geometry score.

Replaces the broken SDF tower (AUC 0.18) with a deterministic
symmetry-asymmetry score (no training). Re-runs the OOD eval and
prints per-source recall/FPR + new ensemble Pareto frontier including
the symmetry channel.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.jepa import IJEPAModel
from src.research.symmetry_geometry import symmetry_score
from scripts.eval_ood_cascade import (
    SEG_ONNX, MIN_TUMOR_AREA, _sess, _preprocess_seg, seg_tta, GT,
)

JEPA_CKPT = ROOT / 'v9b_artifacts' / 'v9b_jepa' / 'last.pt'
SAMPLES_DIR = ROOT / 'samples' / 'ood'

# Detect IXI2D additions (caveat 3 — expand cohort)
EXTRA_HEALTHY_SOURCES = ['healthy_ixi2d']
for ehs in EXTRA_HEALTHY_SOURCES:
    if (SAMPLES_DIR / ehs).exists():
        GT[ehs] = 'no_tumor'


def load_jepa(device):
    ck = torch.load(str(JEPA_CKPT), map_location=device, weights_only=False)
    a = ck.get('args', {})
    m = IJEPAModel(image_size=a.get('image_size', 256), patch_size=16,
                    embed_dim=a.get('embed_dim', 384), depth=a.get('depth', 12),
                    heads=a.get('heads', 6))
    m.load_state_dict(ck['model_state_dict'])
    return m.to(device).eval()


def _stats(rows, key, t):
    TP = sum(1 for r in rows if r['gt']=='tumor' and r[key] > t)
    FN = sum(1 for r in rows if r['gt']=='tumor' and r[key] <= t)
    FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[key] > t)
    TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[key] <= t)
    re = TP/(TP+FN) if TP+FN else 0
    fp = FP/(FP+TN) if FP+TN else 0
    acc = (TP+TN)/(TP+FN+FP+TN) if (TP+FN+FP+TN) else 0
    f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
    return TP, FN, FP, TN, re, fp, acc, f1


def _auc(rows, key):
    pos = [r[key] for r in rows if r['gt']=='tumor' and r[key] is not None]
    neg = [r[key] for r in rows if r['gt']=='no_tumor' and r[key] is not None]
    if not pos or not neg: return float('nan')
    wins = ties = total = 0
    for sp in pos:
        for sn in neg:
            if sp > sn: wins += 1
            elif sp == sn: ties += 1
            total += 1
    return (wins + 0.5*ties) / total


def _best_threshold(rows, key, metric='f1'):
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals: return (None, -1)
    cands = sorted(set(round(v, 4) for v in vals))
    best = (None, -1)
    for t in cands:
        rs = [r for r in rows if r[key] is not None]
        TP, FN, FP, TN, re, fp, acc, f1 = _stats(rs, key, t)
        m = f1 if metric == 'f1' else acc
        if m > best[1]:
            best = (t, m, re, fp, acc, f1)
    return best


def main():
    if not JEPA_CKPT.exists():
        sys.exit(f'missing {JEPA_CKPT}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')
    print(f'[init] loading JEPA ...')
    model = load_jepa(device)

    seg = _sess(SEG_ONNX)
    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png','.jpg','.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples across {len(set(p.parent.name for p in samples))} sources')
    for src in sorted(set(p.parent.name for p in samples)):
        n = sum(1 for p in samples if p.parent.name == src)
        gt = GT[src]
        marker = ' [NEW]' if src in EXTRA_HEALTHY_SOURCES else ''
        print(f'  {src:48s} GT={gt:8s} n={n:3d}{marker}')

    rows = []
    t0 = time.perf_counter()
    last = t0
    for i, p in enumerate(samples):
        img = Image.open(p)
        img_rgb = np.asarray(img.convert('RGB').resize((256, 256), Image.BILINEAR), dtype=np.uint8)
        x = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            emap = model.prediction_error_map(x).squeeze().cpu().numpy()
        jepa_p95 = float(np.percentile(emap, 95))
        sym = symmetry_score(img_rgb, view='axial', percentile=95.0)
        prob = seg_tta(seg, _preprocess_seg(img))
        rec = {
            'source': p.parent.name, 'file': p.name, 'gt': GT[p.parent.name],
            'jepa_p95': jepa_p95,
            'sym_p95': sym,
            'v8_area_020': int((prob >= 0.20).sum()),
        }
        rec['v8_verdict_020'] = 'TUMOR' if rec['v8_area_020'] >= MIN_TUMOR_AREA else 'no_tumor'
        rows.append(rec)
        if time.perf_counter() - last > 20:
            last = time.perf_counter()
            print(f'  [{i+1}/{len(samples)}]  elapsed={time.perf_counter()-t0:.0f}s')
    print(f'\n[done] {(time.perf_counter()-t0)/60:.1f} min\n')

    # ============ AUC ============
    print('='*80)
    print(f'AUC FOR TUMOR-vs-HEALTHY (expanded cohort: {len(rows)} samples)')
    print('='*80)
    for k, name in (('jepa_p95', 'v9b JEPA appearance'),
                     ('sym_p95',  'symmetry geometry (NEW, replaces SDF)')):
        print(f'  {name:40s}  AUC = {_auc(rows, k):.4f}')

    # ============ Best F1 per signal ============
    print('\n' + '='*80)
    print('BEST F1 OPERATING POINT (single signal)')
    print('='*80)
    print(f'  {"signal":30s}  thr      recall    FPR    acc    F1')
    for k in ('jepa_p95', 'sym_p95'):
        best = _best_threshold(rows, k, 'f1')
        if best[0] is None: continue
        t, _, re, fp, acc, f1 = best
        print(f'  {k:30s}  {t:>6.3f}    {re:.0%}     {fp:.0%}     {acc:.0%}    {f1:.2f}')

    # v8 baseline
    TP, FN, FP, TN, re_v8, fp_v8, acc_v8, f1_v8 = _stats(rows, 'v8_area_020', MIN_TUMOR_AREA - 1)
    print(f'  {"v8 segmentation @ 0.20":30s}  ----      {re_v8:.0%}     {fp_v8:.0%}     {acc_v8:.0%}    {f1_v8:.2f}')

    # ============ Per-source ============
    print('\n' + '='*80)
    print('PER-SOURCE BREAKDOWN (recall on tumor / FPR on healthy)')
    print('='*80)
    by_src = {}
    for r in rows: by_src.setdefault(r['source'], []).append(r)
    print(f'{"source":48s} GT     n   v9b_JEPA  symmetry   v8_seg')
    for src in sorted(by_src):
        rs = by_src[src]; gt = rs[0]['gt']; n = len(rs)
        # Use best-F1 threshold per signal
        jepa_t = _best_threshold(rows, 'jepa_p95', 'f1')[0]
        sym_t  = _best_threshold(rows, 'sym_p95', 'f1')[0] if any(r['sym_p95'] is not None for r in rows) else None
        jepa_hits = sum(1 for r in rs if r['jepa_p95'] > jepa_t)
        sym_hits = sum(1 for r in rs if r['sym_p95'] is not None and r['sym_p95'] > (sym_t or 1e9))
        v8_hits = sum(1 for r in rs if r['v8_verdict_020'] == 'TUMOR')
        marker = ' [NEW]' if src in EXTRA_HEALTHY_SOURCES else ''
        print(f'  {src:46s} {gt[:6]:6s} {n:3d}   {jepa_hits/n:>5.0%}    {sym_hits/n:>5.0%}     '
              f'{v8_hits/n:>5.0%}{marker}')

    # ============ Ensemble Pareto frontier ============
    print('\n' + '='*80)
    print('PARETO FRONTIER on EXPANDED cohort (JEPA + symmetry + v8 ensemble)')
    print('='*80)
    from itertools import product
    j_grid = sorted(set(round(r['jepa_p95'], 3) for r in rows))
    s_grid = sorted(set(round(r['sym_p95'], 1) for r in rows if r['sym_p95'] is not None))
    v_grid = [49, 199, 999, 4999]
    rules = {
        'JEPA':       lambda j, s, v: j,
        'symmetry':   lambda j, s, v: s,
        'v8':         lambda j, s, v: v,
        'J | sym':    lambda j, s, v: j or s,
        'J & sym':    lambda j, s, v: j and s,
        'J | v8':     lambda j, s, v: j or v,
        'J | sym | v8': lambda j, s, v: j or s or v,
        '2 of 3':     lambda j, s, v: (int(j)+int(s)+int(v)) >= 2,
    }
    pareto: list[tuple[float, float, str, float, float, int]] = []
    for tj, ts, ta, (name, rfn) in product(j_grid, s_grid, v_grid, rules.items()):
        TP = FN = FP = TN = 0
        for r in rows:
            sym_fires = (r['sym_p95'] is not None) and (r['sym_p95'] > ts)
            fires = rfn(r['jepa_p95'] > tj, sym_fires, r['v8_area_020'] >= ta)
            if r['gt'] == 'tumor':
                TP += fires; FN += (not fires)
            else:
                FP += fires; TN += (not fires)
        re = TP/(TP+FN) if TP+FN else 0
        fp = FP/(FP+TN) if FP+TN else 0
        pareto.append((re, fp, name, tj, ts, ta))

    # Group by recall band, show min FPR per band
    from collections import defaultdict
    by_band = defaultdict(list)
    for re, fp, n, tj, ts, ta in pareto:
        band = round(re * 20) / 20
        by_band[band].append((fp, n, tj, ts, ta, re))
    print(f'  {"recall_band":12s}  min_FPR  best_rule       thresholds')
    for band in sorted(by_band, reverse=True):
        items = sorted(by_band[band])
        fp, n, tj, ts, ta, re_actual = items[0]
        print(f'  recall>={band*100:.0f}%    {fp*100:>5.1f}%   {n:14s}  J>{tj:.3f} '
              f'S>{ts:>5.1f} v8>={ta:>5d}  (actual={re_actual:.0%})')

    # ============ Save ============
    out_csv = SAMPLES_DIR / 'eval_v9b_symmetry_expanded.csv'
    fields = list(rows[0].keys())
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
