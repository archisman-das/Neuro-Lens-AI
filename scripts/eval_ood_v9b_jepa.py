"""Thorough OOD evaluation of v9b normative JEPA + conformal calibration.

Pipeline per image:
  1. Resize to 256x256, RGB, [0,1]
  2. JEPA prediction_error_map(x) — for each of the 256 patches, mask it
     out, predict its latent from the other 255, measure smooth-L1
     residual. Returns (256, 256) anomaly map.
  3. Per-image score = 95th percentile of all pixel residuals (matches
     what conformal calibration used).
  4. Conformal verdict: per_image_score > q  ->  anomaly (tumor).
     q=0.308 loaded from v9b_conformal.json (alpha=0.10, empirical
     coverage 0.90 on 17,487 healthy calibration samples).

Compares v9b verdict against:
  - v8 segmentation alone @ threshold 0.20 (the current production baseline)
  - the 4 historical classifier sets (OLD, v8-RAW, v8-BAL, v8-MVMM)
    via per-classifier accuracy already in our earlier audit CSVs.

Outputs:
  - per-image table (csv)
  - per-source aggregate
  - 4-policy comparison table for the executive summary
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

from src.research.jepa import IJEPAModel
from scripts.eval_ood_cascade import (
    SEG_ONNX, MIN_TUMOR_AREA, _sess, _preprocess_seg, seg_tta, GT,
)

JEPA_CKPT = ROOT / 'v9b_artifacts' / 'v9b_jepa' / 'last.pt'
CONFORMAL_JSON = ROOT / 'v9b_artifacts' / 'v9b_conformal.json'
SAMPLES_DIR = ROOT / 'samples' / 'ood'
IMAGE_SIZE = 256


def load_jepa(device: str = 'cuda') -> IJEPAModel:
    ckpt = torch.load(str(JEPA_CKPT), map_location=device, weights_only=False)
    a = ckpt.get('args') or {}
    model = IJEPAModel(
        image_size=a.get('image_size', 256),
        patch_size=a.get('patch_size', 16),
        in_chans=a.get('in_chans', 3),
        embed_dim=a.get('embed_dim', 384),
        depth=a.get('depth', 12),
        heads=a.get('heads', 6),
    )
    sd = ckpt.get('model_state_dict') or ckpt
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f'  [warn] checkpoint load: missing={len(miss)} unexpected={len(unexp)}')
    model = model.to(device).eval()
    return model


def jepa_anomaly_map(model: IJEPAModel, img_pil: Image.Image, device: str = 'cuda') -> np.ndarray:
    img = img_pil.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        emap = model.prediction_error_map(x)  # (1, 1, IMAGE_SIZE, IMAGE_SIZE)
    return emap.squeeze().cpu().numpy()


def main():
    if not JEPA_CKPT.exists():
        sys.exit(f'missing {JEPA_CKPT}')
    if not CONFORMAL_JSON.exists():
        sys.exit(f'missing {CONFORMAL_JSON}')
    conformal = json.loads(CONFORMAL_JSON.read_text())
    q = float(conformal['q'])
    alpha = float(conformal['alpha'])
    print(f'[init] conformal q={q:.4f} alpha={alpha} (target coverage = {1-alpha:.0%})')
    print(f'[init] calibrated on {conformal["report"]["n_calib"]} healthy samples, '
          f'empirical_coverage={conformal["report"]["empirical_coverage"]:.4f}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))
    model = load_jepa(device)
    seg = _sess(SEG_ONNX)

    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples\n')

    rows = []
    t0 = time.perf_counter()
    last_print = t0
    for i, p in enumerate(samples):
        img = Image.open(p)
        gt = GT[p.parent.name]

        # v9b JEPA + conformal
        emap = jepa_anomaly_map(model, img, device)
        p95 = float(np.percentile(emap, 95))
        max_err = float(emap.max())
        mean_err = float(emap.mean())
        # Anomaly mask (pixels with error > q)
        ano_mask = (emap > q).astype(np.uint8)
        ano_area = int(ano_mask.sum())
        v9b_verdict = 'TUMOR' if p95 > q else 'no_tumor'

        # v8 segmentation comparison
        prob = seg_tta(seg, _preprocess_seg(img))
        v8_area = int((prob >= 0.20).sum())
        v8_verdict = 'TUMOR' if v8_area >= MIN_TUMOR_AREA else 'no_tumor'

        rows.append({
            'source': p.parent.name,
            'file': p.name,
            'gt': gt,
            'v9b_p95': round(p95, 4),
            'v9b_max': round(max_err, 4),
            'v9b_mean': round(mean_err, 4),
            'v9b_ano_area': ano_area,
            'v9b_verdict': v9b_verdict,
            'v8_area_020': v8_area,
            'v8_verdict': v8_verdict,
        })
        if time.perf_counter() - last_print > 20:
            last_print = time.perf_counter()
            print(f'  [{i+1}/{len(samples)}]  elapsed={time.perf_counter()-t0:.0f}s')

    elapsed = time.perf_counter() - t0
    print(f'\n[done] {len(rows)} samples in {elapsed:.0f}s ({elapsed/len(rows):.2f}/sample)\n')

    # ======================== aggregate ============================
    def _stats(rs, col):
        TP = sum(1 for r in rs if r['gt']=='tumor' and r[col]=='TUMOR')
        FN = sum(1 for r in rs if r['gt']=='tumor' and r[col]=='no_tumor')
        FP = sum(1 for r in rs if r['gt']=='no_tumor' and r[col]=='TUMOR')
        TN = sum(1 for r in rs if r['gt']=='no_tumor' and r[col]=='no_tumor')
        recall = TP/(TP+FN) if TP+FN else 0
        fpr = FP/(FP+TN) if FP+TN else 0
        acc = (TP+TN)/(TP+FN+FP+TN)
        f1 = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0
        return TP, FN, FP, TN, recall, fpr, acc, f1

    print('='*82)
    print('v9b NORMATIVE JEPA + CONFORMAL — OOD SCORECARD')
    print('='*82)
    for col, label in (('v9b_verdict', 'v9b JEPA + conformal (q=0.308)'),
                        ('v8_verdict', 'v8 segmentation @ 0.20 (baseline)')):
        TP, FN, FP, TN, re, fp, acc, f1 = _stats(rows, col)
        print(f'\n  {label}')
        print(f'    TP={TP:2d}  FN={FN:2d}  FP={FP:2d}  TN={TN:2d}')
        print(f'    recall={re:.0%}   FPR={fp:.0%}   accuracy={acc:.0%}   F1={f1:.2f}')

    # ======================== per-source ===========================
    print('\n' + '='*82)
    print('PER-SOURCE BREAKDOWN')
    print('='*82)
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    print(f'\n{"source":48s} GT     n   v9b_recall/FPR   v8_recall/FPR')
    for src in sorted(by_src):
        rs = by_src[src]; gt = rs[0]['gt']; n = len(rs)
        v9b_hits = sum(1 for r in rs if r['v9b_verdict']=='TUMOR')
        v8_hits  = sum(1 for r in rs if r['v8_verdict']=='TUMOR')
        if gt == 'tumor':
            print(f'  {src:46s} pos    {n:3d}   recall={v9b_hits/n:.0%}'.ljust(76)
                  + f'     recall={v8_hits/n:.0%}')
        else:
            print(f'  {src:46s} neg    {n:3d}   FPR={v9b_hits/n:.0%}'.ljust(76)
                  + f'        FPR={v8_hits/n:.0%}')

    # ======================== 4-policy comparison =================
    print('\n' + '='*82)
    print('EXECUTIVE SUMMARY — every policy we have measured to date')
    print('='*82)
    # Historical numbers from the audit scripts we already ran
    OLD = {'recall_range': '28-42%', 'fpr_range': '8-58%'}
    BAL = {'recall_range': '31-44%', 'fpr_range': '0%'}
    MVMM = {'recall_range': '25-47%', 'fpr_range': '0%'}
    TP, FN, FP, TN, re_v9b, fp_v9b, acc_v9b, f1_v9b = _stats(rows, 'v9b_verdict')
    _, _, _, _, re_v8, fp_v8, acc_v8, _ = _stats(rows, 'v8_verdict')
    print(f'\n  {"policy":40s}  recall      FPR        accuracy')
    print(f'  {"OLD 3 classifiers (Kaggle-only)":40s}  {OLD["recall_range"]:>10s}  {OLD["fpr_range"]:>9s}  31-54%')
    print(f'  {"v8-BAL 3 classifiers (+OpenNeuro)":40s}  {BAL["recall_range"]:>10s}  {BAL["fpr_range"]:>9s}  46-58%')
    print(f'  {"v8-MVMM 3 classifiers (+multi-view)":40s}  {MVMM["recall_range"]:>10s}  {MVMM["fpr_range"]:>9s}  44-60%')
    print(f'  {"v8 segmentation only @ 0.20":40s}  {re_v8:>10.0%}  {fp_v8:>9.0%}  {acc_v8:>8.0%}')
    print(f'  {"v9b JEPA + conformal":40s}  {re_v9b:>10.0%}  {fp_v9b:>9.0%}  {acc_v9b:>8.0%}   <-- NEW')

    # ======================== persist ==============================
    out_csv = SAMPLES_DIR / 'eval_v9b_jepa_results.csv'
    fields = list(rows[0].keys())
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
