"""Phase 5 — v9b ANDi DDPM (pyramidal-noise + unconditional) OOD eval.

Loads the trained ANDi DDPM (Frotscher et al. 2024 setup: unconditional
+ pyramidal noise during training, standard Gaussian noise at inference),
computes per-image anomaly scores via andi_anomaly_map() aggregated over
timesteps [75, 200] stride 5, and reports AUC + best-F1 on the
246-sample expanded OOD bench.

If standalone AUC >= 0.7 we'll roll into the 4-signal ensemble
(v9c + v8 + sym + ANDi) and search for a 95/10 rule.
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

from src.research.andi_inference import andi_anomaly_map
from src.research.latent_diffusion_decoder import LatentConditionedDDPM
from scripts.eval_ood_cascade import GT as _GT

GT = dict(_GT)
GT.setdefault('healthy_ixi2d', 'no_tumor')
GT.setdefault('healthy_navoneel', 'no_tumor')

SOURCE_GROUPS = {
    'tumor_binary_navoneel_via_miladfa7': 'navoneel',
    'healthy_navoneel': 'navoneel',
}
def _source_group(folder: str) -> str:
    return SOURCE_GROUPS.get(folder, folder)


CKPT = ROOT / 'v9b_artifacts' / 'v9b_andi_ddpm' / 'last.pt'
SAMPLES = ROOT / 'samples' / 'ood'


def _preprocess(img: Image.Image, image_size: int = 256) -> torch.Tensor:
    arr = np.asarray(img.convert('RGB').resize((image_size, image_size), Image.BILINEAR),
                      dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')

    if not CKPT.exists():
        sys.exit(f'ERROR: checkpoint missing at {CKPT}')

    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    a = ck.get('args', {})
    cond_dim = a.get('cond_dim', 384)
    image_size = a.get('image_size', 256)
    print(f'[init] cond_dim={cond_dim} image_size={image_size} '
          f'epoch={ck.get("epoch")}  desc="{ck.get("description")}"')

    ddpm = LatentConditionedDDPM(in_chans=3, base_ch=32, cond_dim=cond_dim).to(device)
    miss, unexp = ddpm.load_state_dict(ck['model_state_dict'], strict=False)
    print(f'  loaded DDPM: missing={len(miss)} unexpected={len(unexp)}')
    ddpm.eval()

    samples = sorted(p for p in SAMPLES.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples')

    rows = []
    t0 = time.perf_counter()
    last = t0
    for i, p in enumerate(samples):
        img = Image.open(p)
        x0 = _preprocess(img, image_size).unsqueeze(0).to(device)
        cond = torch.zeros(1, cond_dim, device=device)  # unconditional
        with torch.no_grad():
            amap = andi_anomaly_map(ddpm, x0, cond, t_low=75, t_high=200,
                                     stride=5, device=device, seed=0)
        flat = amap.flatten().cpu().numpy()
        rows.append({
            'source': p.parent.name, 'file': p.name,
            'gt': GT[p.parent.name],
            'source_group': _source_group(p.parent.name),
            'p95': float(np.percentile(flat, 95)),
            'p99': float(np.percentile(flat, 99)),
            'max': float(flat.max()),
            'mean': float(flat.mean()),
        })
        if time.perf_counter() - last > 30:
            last = time.perf_counter()
            rate = (i + 1) / (time.perf_counter() - t0)
            eta = (len(samples) - i - 1) / max(rate, 1e-6)
            print(f'  [{i+1}/{len(samples)}]  elapsed={time.perf_counter()-t0:.0f}s  '
                  f'rate={rate:.2f}/s  eta={eta:.0f}s')
    print(f'\n[done] {len(rows)} samples in {(time.perf_counter()-t0)/60:.1f} min')

    # AUC overall + per-source-group
    def _auc(rows, key='p95'):
        pos = [r[key] for r in rows if r['gt'] == 'tumor']
        neg = [r[key] for r in rows if r['gt'] == 'no_tumor']
        if not pos or not neg:
            return float('nan')
        wins = ties = total = 0
        for sp in pos:
            for sn in neg:
                if sp > sn: wins += 1
                elif sp == sn: ties += 1
                total += 1
        return (wins + 0.5 * ties) / total

    print('\n  ANDi standalone AUC (each percentile):')
    for k in ('p95', 'p99', 'max', 'mean'):
        print(f'    {k:>4s} = {_auc(rows, k):.4f}')

    navoneel = [r for r in rows if r['source_group'] == 'navoneel']
    print(f'\n  ANDi LOSO AUC (Navoneel only, n={len(navoneel)}): '
          f'p95={_auc(navoneel, "p95"):.4f}')

    # Best-F1 sweep on p95
    def _stats(rows, t):
        TP = sum(1 for r in rows if r['gt']=='tumor' and r['p95'] > t)
        FN = sum(1 for r in rows if r['gt']=='tumor' and r['p95'] <= t)
        FP = sum(1 for r in rows if r['gt']=='no_tumor' and r['p95'] > t)
        TN = sum(1 for r in rows if r['gt']=='no_tumor' and r['p95'] <= t)
        re = TP/(TP+FN) if TP+FN else 0
        fp = FP/(FP+TN) if FP+TN else 0
        pr = TP/(TP+FP) if TP+FP else 0
        acc = (TP+TN)/len(rows) if rows else 0
        f1 = 2*pr*re/(pr+re) if pr+re else 0
        return TP, FN, FP, TN, re, fp, pr, acc, f1

    print('\n  ANDi PARETO FRONTIER (sweep p95 threshold):')
    print(f'  {"recall":>7s}  {"min_FPR":>7s}  {"threshold":>10s}  {"F1":>5s}')
    thresholds = sorted(set(round(r['p95'], 6) for r in rows))
    band_best = {}
    for t in thresholds:
        TP, FN, FP, TN, re, fp, pr, acc, f1 = _stats(rows, t)
        band = round(re * 20) / 20
        if band not in band_best or fp < band_best[band][0]:
            band_best[band] = (fp, t, f1, acc)
    for band in sorted(band_best, reverse=True):
        fp, t, f1, acc = band_best[band]
        print(f'  {band*100:>5.0f}%   {fp*100:>5.1f}%   {t:>9.6g}   {f1:.2f}')

    best_t, best_f1 = None, -1
    for t in thresholds:
        _, _, _, _, re, fp, pr, _, f1 = _stats(rows, t)
        if f1 > best_f1:
            best_t, best_f1, best_re, best_fp, best_pr = t, f1, re, fp, pr
    print(f'\n  BEST F1: t={best_t:.6g}  recall={best_re:.0%}  FPR={best_fp:.0%}  '
          f'prec={best_pr:.0%}  F1={best_f1:.3f}')

    # Save CSV (ensemble script will pick this up)
    out_csv = SAMPLES / 'eval_v9b_andi_results.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
