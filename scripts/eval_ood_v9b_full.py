"""Full v9b OOD eval: JEPA + DDPM + SDF + two-tower combine.

Compares 4 v9b score variants against the v8 segmentation baseline:
  - v9b_app:     JEPA appearance anomaly only (p95 of prediction_error_map)
  - v9b_geo:     SDF geometric anomaly only (p95 of SDF deviation)
  - v9b_combo:   two-tower weighted_sum (lambda_app=0.6, lambda_geo=0.4)
  - v9b_residual: DDPM healthy-counterfactual residual (|x - x_healthy|)
                  (much slower because of DDIM sampling)

For each variant, per-image p95 score → tumor/no_tumor verdict at the
threshold sweep optimum. AUC computed against ground truth.
"""
from __future__ import annotations

import csv
import json
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

JEPA_CKPT = ROOT / 'v9b_artifacts' / 'v9b_jepa' / 'last.pt'
STAGE2_CKPT = ROOT / 'v9b_artifacts' / 'v9b_stage2' / 'last.pt'
CONFORMAL_JSON = ROOT / 'v9b_artifacts' / 'v9b_conformal.json'
SAMPLES_DIR = ROOT / 'samples' / 'ood'
IMAGE_SIZE = 256


def preprocess(img_pil: Image.Image, device: str) -> torch.Tensor:
    img = img_pil.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _stats(rows, score_key, t):
    TP = sum(1 for r in rows if r['gt']=='tumor' and r[score_key] > t)
    FN = sum(1 for r in rows if r['gt']=='tumor' and r[score_key] <= t)
    FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[score_key] > t)
    TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[score_key] <= t)
    recall = TP/(TP+FN) if TP+FN else 0
    fpr = FP/(FP+TN) if FP+TN else 0
    acc = (TP+TN)/len(rows)
    f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
    return TP, FN, FP, TN, recall, fpr, acc, f1


def _auc(rows, score_key):
    pos = [r[score_key] for r in rows if r['gt']=='tumor']
    neg = [r[score_key] for r in rows if r['gt']=='no_tumor']
    if not pos or not neg: return float('nan')
    wins = ties = total = 0
    for sp in pos:
        for sn in neg:
            if sp > sn: wins += 1
            elif sp == sn: ties += 1
            total += 1
    return (wins + 0.5*ties) / total


def _best_threshold(rows, score_key, metric='f1'):
    candidates = sorted(set(round(r[score_key], 4) for r in rows))
    best = (None, -1.0)
    for t in candidates:
        TP, FN, FP, TN, re, fp, acc, f1 = _stats(rows, score_key, t)
        m = f1 if metric == 'f1' else acc
        if m > best[1]:
            best = (t, m, re, fp, acc, f1)
    return best


def main():
    if not all(p.exists() for p in (JEPA_CKPT, STAGE2_CKPT, CONFORMAL_JSON)):
        sys.exit('missing one of the v9b artefacts')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))

    print('[init] loading V9BModel (JEPA + DDPM + SDF + conformal) ...')
    t0 = time.perf_counter()
    model = V9BModel.from_checkpoints(
        str(JEPA_CKPT), str(STAGE2_CKPT), str(CONFORMAL_JSON),
        image_size=IMAGE_SIZE, device=device,
    )
    print(f'  loaded in {time.perf_counter()-t0:.1f}s'
          f' (JEPA={model.jepa is not None}, DDPM={model.ddpm is not None}, '
          f'SDF={model.sdf_tower is not None}, conformal_q={model.conformal.q:.4f})')

    seg = _sess(SEG_ONNX)
    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png','.jpg','.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples')

    # Decide if we run the DDPM residual (50 DDIM steps per image, slow).
    # Default: skip to save time; turn on with V9B_RUN_DDPM=1
    import os
    run_ddpm = os.environ.get('V9B_RUN_DDPM', '0').strip() == '1'
    print(f'[init] DDPM residual: {"ON (slow, ~30s/sample)" if run_ddpm else "OFF (set V9B_RUN_DDPM=1 to enable)"}')
    print()

    rows = []
    t0 = time.perf_counter()
    last = t0
    for i, p in enumerate(samples):
        img = Image.open(p)
        x = preprocess(img, device)
        out = model.infer(x, combine_mode='weighted_sum',
                           lambda_app=0.6, lambda_geo=0.4,
                           ddpm_num_steps=50 if run_ddpm else 0)
        # Per-image scores: 95th percentile of each anomaly map
        app = out['appearance_anomaly'].squeeze().cpu().numpy()
        geo = out['geometry_anomaly'].squeeze().cpu().numpy() if out['geometry_anomaly'] is not None else None
        combo = out['combined_anomaly'].squeeze().cpu().numpy()
        rec = {
            'source': p.parent.name, 'file': p.name, 'gt': GT[p.parent.name],
            'v9b_app_p95':    float(np.percentile(app, 95)),
            'v9b_app_max':    float(app.max()),
            'v9b_geo_p95':    float(np.percentile(geo, 95)) if geo is not None else 0.0,
            'v9b_combo_p95':  float(np.percentile(combo, 95)),
            'v9b_combo_max':  float(combo.max()),
        }
        if run_ddpm and out['residual'] is not None:
            res = out['residual'].squeeze().cpu().numpy()
            rec['v9b_residual_p95'] = float(np.percentile(res, 95))
            rec['v9b_residual_mean'] = float(res.mean())
        # v8 segmentation baseline
        prob = seg_tta(seg, _preprocess_seg(img))
        rec['v8_area_020'] = int((prob >= 0.20).sum())
        rec['v8_verdict_020'] = 'TUMOR' if rec['v8_area_020'] >= MIN_TUMOR_AREA else 'no_tumor'
        rows.append(rec)
        if time.perf_counter() - last > 20:
            last = time.perf_counter()
            print(f'  [{i+1}/{len(samples)}]  elapsed={time.perf_counter()-t0:.0f}s')
    print(f'\n[done] {len(rows)} samples in {(time.perf_counter()-t0)/60:.1f} min\n')

    # ===================== AUC for each score variant =====================
    print('='*82)
    print('AUC for tumor-vs-healthy on OOD per scoring variant')
    print('='*82)
    score_keys = ['v9b_app_p95', 'v9b_geo_p95', 'v9b_combo_p95']
    if run_ddpm:
        score_keys.append('v9b_residual_p95')
    for k in score_keys:
        print(f'  AUC({k}) = {_auc(rows, k):.4f}')

    # ===================== best operating point per variant =================
    print('\n' + '='*82)
    print('BEST F1 OPERATING POINT per scoring variant')
    print('='*82)
    print(f'  {"variant":18s}  thr      recall    FPR    acc    F1')
    for k in score_keys:
        t, _, re, fp, acc, f1 = _best_threshold(rows, k, 'f1')
        print(f'  {k:18s}  {t:.3f}    {re:.0%}    {fp:.0%}   {acc:.0%}   {f1:.2f}')

    # v8 baseline for reference
    TP, FN, FP, TN, re_v8, fp_v8, acc_v8, f1_v8 = _stats(rows, 'v8_area_020', MIN_TUMOR_AREA - 1)
    print(f'  {"v8 seg @ 0.20":18s}  ----     {re_v8:.0%}    {fp_v8:.0%}   {acc_v8:.0%}   {f1_v8:.2f}')

    # ===================== Pareto curve for combined score =================
    print('\n' + '='*82)
    print('THRESHOLD SWEEP — v9b_combo_p95 (best variant)')
    print('='*82)
    print(f'  {"thr":>6s}   recall   FPR    accuracy  F1')
    candidates = sorted(set(round(r['v9b_combo_p95'], 3) for r in rows))
    for t in candidates:
        TP, FN, FP, TN, re, fp, acc, f1 = _stats(rows, 'v9b_combo_p95', t)
        print(f'  {t:>6.3f}   {re:>5.0%}   {fp:>5.0%}   {acc:>6.0%}  {f1:.2f}')

    # ===================== per-source on best combo =====================
    print('\n' + '='*82)
    print('PER-SOURCE on v9b_combo @ best-F1 threshold')
    print('='*82)
    best_t = _best_threshold(rows, 'v9b_combo_p95', 'f1')[0]
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    print(f'\nthreshold = {best_t:.3f}')
    print(f'{"source":48s} GT     n   v9b_combo_recall/FPR   v8_recall/FPR')
    for src in sorted(by_src):
        rs = by_src[src]; gt = rs[0]['gt']; n = len(rs)
        v9b_hits = sum(1 for r in rs if r['v9b_combo_p95'] > best_t)
        v8_hits  = sum(1 for r in rs if r['v8_verdict_020'] == 'TUMOR')
        kind = 'recall' if gt=='tumor' else 'FPR'
        print(f'  {src:46s} {gt[:6]:6s} {n:3d}   v9b_{kind}={v9b_hits/n:.0%}'.ljust(80)
              + f'    v8_{kind}={v8_hits/n:.0%}')

    # ===================== final scoreboard =====================
    print('\n' + '='*82)
    print('FINAL SCOREBOARD — every policy on this OOD test bench')
    print('='*82)
    rows_final = []
    rows_final.append(('OLD 3 classifiers (Kaggle-only)', '28-42%', '8-58%', '31-54%'))
    rows_final.append(('v8-MVMM 3 classifiers (multi-view)', '25-47%', '0%', '44-60%'))
    rows_final.append((f'v8 segmentation only @ 0.20',
                       f'{re_v8:.0%}', f'{fp_v8:.0%}', f'{acc_v8:.0%}'))
    for k, label in (('v9b_app_p95', 'v9b JEPA appearance (Stage 1 only)'),
                     ('v9b_geo_p95', 'v9b SDF geometry tower (Stage 2 only)'),
                     ('v9b_combo_p95', 'v9b two-tower combo (full stack)')):
        t, _, re, fp, acc, f1 = _best_threshold(rows, k, 'f1')
        rows_final.append((label, f'{re:.0%}', f'{fp:.0%}', f'{acc:.0%}'))
    print(f'\n  {"policy":42s}  recall      FPR        accuracy')
    for label, r, f, a in rows_final:
        print(f'  {label:42s}  {r:>9s}  {f:>9s}  {a:>9s}')

    # ===================== persist =====================
    out_csv = SAMPLES_DIR / 'eval_v9b_full_results.csv'
    fields = list(rows[0].keys())
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
