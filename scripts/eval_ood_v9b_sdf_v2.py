"""Re-run v9b full eval with the retrained SDF tower (sdf_v2).

Loads V9BModel from original Stage 2 ckpt, then OVERRIDES the SDF tower
weights with v9b_stage2_sdf_v2/last.pt. Same OOD eval as
eval_ood_v9b_full.py — direct apples-to-apples comparison.

Reports new AUC for v9b_geo / v9b_combo to confirm whether the per-image
SDF target moved the geometry tower from the AUC=0.10 (anti-correlated)
zone toward something useful.
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

from src.research.v9b_model import V9BModel
from scripts.eval_ood_cascade import (
    SEG_ONNX, MIN_TUMOR_AREA, _sess, _preprocess_seg, seg_tta, GT,
)

JEPA_CKPT     = ROOT / 'v9b_artifacts' / 'v9b_jepa'     / 'last.pt'
STAGE2_CKPT   = ROOT / 'v9b_artifacts' / 'v9b_stage2'   / 'last.pt'
SDF_V2_CKPT   = ROOT / 'v9b_artifacts' / 'v9b_stage2_sdf_v2' / 'last.pt'
CONFORMAL     = ROOT / 'v9b_artifacts' / 'v9b_conformal.json'
SAMPLES_DIR   = ROOT / 'samples' / 'ood'
IMAGE_SIZE    = 256


def preprocess(img_pil, device):
    img = img_pil.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _stats(rows, key, t):
    TP = sum(1 for r in rows if r['gt']=='tumor' and r[key] > t)
    FN = sum(1 for r in rows if r['gt']=='tumor' and r[key] <= t)
    FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[key] > t)
    TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[key] <= t)
    re = TP/(TP+FN) if TP+FN else 0
    fp = FP/(FP+TN) if FP+TN else 0
    acc = (TP+TN)/len(rows)
    f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
    return TP, FN, FP, TN, re, fp, acc, f1


def _auc(rows, key):
    pos = [r[key] for r in rows if r['gt']=='tumor']
    neg = [r[key] for r in rows if r['gt']=='no_tumor']
    if not pos or not neg: return float('nan')
    wins = ties = total = 0
    for sp in pos:
        for sn in neg:
            if sp > sn: wins += 1
            elif sp == sn: ties += 1
            total += 1
    return (wins + 0.5*ties) / total


def _best_threshold(rows, key, metric='f1'):
    cands = sorted(set(round(r[key], 4) for r in rows))
    best = (None, -1)
    for t in cands:
        TP, FN, FP, TN, re, fp, acc, f1 = _stats(rows, key, t)
        m = f1 if metric == 'f1' else acc
        if m > best[1]:
            best = (t, m, re, fp, acc, f1)
    return best


def main():
    for p in (JEPA_CKPT, STAGE2_CKPT, SDF_V2_CKPT, CONFORMAL):
        if not p.exists():
            sys.exit(f'missing {p}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')
    print(f'[init] loading V9BModel ...')
    model = V9BModel.from_checkpoints(str(JEPA_CKPT), str(STAGE2_CKPT),
                                       str(CONFORMAL), image_size=IMAGE_SIZE, device=device)
    print(f'  loaded (JEPA={model.jepa is not None}, '
          f'DDPM={model.ddpm is not None}, SDF={model.sdf_tower is not None})')

    # SWAP IN the retrained SDF tower weights
    sdf_ck = torch.load(str(SDF_V2_CKPT), map_location=device, weights_only=False)
    sdf_sd = sdf_ck.get('sdf_state_dict', sdf_ck)
    miss, unexp = model.sdf_tower.load_state_dict(sdf_sd, strict=False)
    print(f'  SDF v2 swapped in (missing={len(miss)} unexpected={len(unexp)})')

    seg = _sess(SEG_ONNX)
    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png','.jpg','.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples\n')

    rows = []
    t0 = time.perf_counter()
    last = t0
    for i, p in enumerate(samples):
        img = Image.open(p)
        x = preprocess(img, device)
        out = model.infer(x, combine_mode='weighted_sum',
                           lambda_app=0.6, lambda_geo=0.4, ddpm_num_steps=0)
        app = out['appearance_anomaly'].squeeze().cpu().numpy()
        geo = out['geometry_anomaly'].squeeze().cpu().numpy() if out['geometry_anomaly'] is not None else None
        combo = out['combined_anomaly'].squeeze().cpu().numpy()
        rec = {
            'source': p.parent.name, 'file': p.name, 'gt': GT[p.parent.name],
            'v9b_app_p95':   float(np.percentile(app, 95)),
            'v9b_geo_p95':   float(np.percentile(geo, 95)) if geo is not None else 0.0,
            'v9b_combo_p95': float(np.percentile(combo, 95)),
        }
        prob = seg_tta(seg, _preprocess_seg(img))
        rec['v8_area_020'] = int((prob >= 0.20).sum())
        rec['v8_verdict_020'] = 'TUMOR' if rec['v8_area_020'] >= MIN_TUMOR_AREA else 'no_tumor'
        rows.append(rec)
        if time.perf_counter() - last > 20:
            last = time.perf_counter()
            print(f'  [{i+1}/{len(samples)}]  elapsed={time.perf_counter()-t0:.0f}s')
    print(f'\n[done] {(time.perf_counter()-t0)/60:.1f} min\n')

    # ============ AUC comparison ============
    print('='*84)
    print('AUC: SDF v2 vs original SDF (recall on OOD tumor-vs-healthy)')
    print('='*84)
    OLD_AUC = {'app': 0.857, 'geo': 0.100, 'combo': 0.333}  # from previous eval
    for key, old_auc in (('v9b_app_p95', OLD_AUC['app']),
                         ('v9b_geo_p95', OLD_AUC['geo']),
                         ('v9b_combo_p95', OLD_AUC['combo'])):
        new_auc = _auc(rows, key)
        delta = new_auc - old_auc
        flag = ' [BIG WIN]' if delta > 0.20 else (' [improved]' if delta > 0.05 else (' [same]' if abs(delta) <= 0.05 else ' [worse]'))
        print(f'  {key:18s}  old={old_auc:.3f}  new={new_auc:.3f}  delta={delta:+.3f}{flag}')

    # ============ best F1 per variant ============
    print('\n' + '='*84)
    print('BEST F1 OPERATING POINT (with new SDF v2)')
    print('='*84)
    print(f'  {"variant":18s}  thr      recall    FPR    acc    F1')
    for key in ('v9b_app_p95', 'v9b_geo_p95', 'v9b_combo_p95'):
        t, _, re, fp, acc, f1 = _best_threshold(rows, key, 'f1')
        print(f'  {key:18s}  {t:.3f}    {re:.0%}    {fp:.0%}   {acc:.0%}   {f1:.2f}')
    # v8 baseline for reference
    TP, FN, FP, TN, re_v8, fp_v8, acc_v8, f1_v8 = _stats(rows, 'v8_area_020', MIN_TUMOR_AREA - 1)
    print(f'  {"v8 seg @ 0.20":18s}  ----     {re_v8:.0%}    {fp_v8:.0%}   {acc_v8:.0%}   {f1_v8:.2f}')

    # ============ per-source ============
    print('\n' + '='*84)
    print('PER-SOURCE on v9b_combo @ best-F1 threshold (with SDF v2)')
    print('='*84)
    best_t = _best_threshold(rows, 'v9b_combo_p95', 'f1')[0]
    by_src = {}
    for r in rows: by_src.setdefault(r['source'], []).append(r)
    print(f'\nthreshold = {best_t:.3f}')
    print(f'{"source":48s} GT     n   combo  app   geo')
    for src in sorted(by_src):
        rs = by_src[src]; gt = rs[0]['gt']; n = len(rs)
        kind = 'recall' if gt=='tumor' else 'FPR'
        combo_hits = sum(1 for r in rs if r['v9b_combo_p95'] > best_t)
        app_t = _best_threshold(rows, 'v9b_app_p95', 'f1')[0]
        app_hits = sum(1 for r in rs if r['v9b_app_p95'] > app_t)
        geo_t = _best_threshold(rows, 'v9b_geo_p95', 'f1')[0]
        geo_hits = sum(1 for r in rs if r['v9b_geo_p95'] > geo_t)
        print(f'  {src:46s} {gt[:6]:6s} {n:3d}   {combo_hits/n:.0%}    '
              f'{app_hits/n:.0%}    {geo_hits/n:.0%}    <- {kind}')

    # ============ persist ============
    out_csv = SAMPLES_DIR / 'eval_v9b_sdf_v2_results.csv'
    fields = list(rows[0].keys())
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
