"""Proper 3D evaluation of CrossJEPA Method 1.

  Part A: PER-VOLUME anomaly score = MEAN over N axial slices.
          Healthy (radiata-ai, in-distribution for the model) vs
          Tumor (BraTS-2021, out-of-distribution).
          Reports AUC + Pareto + best-F1.

  Part B: PER-SLICE scoring + AGGREGATION SWEEP. For each volume,
          score every axial slice in the central 60%, then aggregate
          via {mean, max, p95, top-k mean}. Compares which aggregation
          gives the cleanest healthy-vs-tumor separation.

HONEST CAVEAT (acknowledged in the output): the healthy radiata-ai
volumes used here were ALL in the training set, so the healthy scores
are an optimistic lower bound. A proper held-out healthy test would
require pulling additional healthy MRI data (e.g. IXI / openneuro NIfTI)
that the model has never seen. Even so, the tumor-healthy gap on this
config is large enough that the relative ordering still tells us
whether the model learned a real anomaly signal.
"""
from __future__ import annotations

import csv
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.v9c_crossjepa.dataset_3d import Vol2SliceDataset
from src.research.v9c_crossjepa.v8_teacher import V8FrozenTeacher
from src.research.v9c_crossjepa.volume_to_slice import Vol2SliceModel


HEALTHY_GLOB = ('c:/Users/anish/.cache/huggingface/hub/'
                 'datasets--radiata-ai--brain-structure/snapshots/'
                 'aea73e2a9734955407eb8bcf2325b479a5f39144/'
                 '*/sub-*/ses-*/anat/*.nii.gz')
BRATS_ROOT = Path('d:/datasets/brats2021')
CKPT = ROOT / 'v9b_artifacts' / 'v9c_crossjepa_method1' / 'last.pt'
V8_PT = ROOT / 'model' / 'best_micro.pt'
OUT_CSV = ROOT / 'samples' / 'ood' / 'eval_method1_3d_results.csv'

# Sample sizes — keep small enough to finish in a few minutes on the
# local 8 GB GPU. ~200 vols * 8 slices * ~0.1s/slice = ~3 min.
N_HEALTHY = 200
N_TUMOR = 200
N_SLICES_PART_A = 8           # mean over 8 slices = per-volume score
N_SLICES_PART_B = 16          # denser sample for aggregation sweep
N_TUMOR_PART_B = 50           # part B is more expensive; smaller cohort
N_HEALTHY_PART_B = 50


def _auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    wins = ties = total = 0
    for sp in pos:
        for sn in neg:
            if sp > sn: wins += 1
            elif sp == sn: ties += 1
            total += 1
    return (wins + 0.5 * ties) / total


def _pareto_report(rows, score_key):
    """rows = list of dicts with 'gt' in {'tumor','no_tumor'} + score_key.
    Prints Pareto frontier + best-F1."""
    pos = [r[score_key] for r in rows if r['gt'] == 'tumor']
    neg = [r[score_key] for r in rows if r['gt'] == 'no_tumor']
    auc = _auc(pos, neg)
    print(f'    AUC = {auc:.4f}  (tumor n={len(pos)}, healthy n={len(neg)})')

    def stats_at(t):
        TP = sum(1 for v in pos if v > t)
        FN = len(pos) - TP
        FP = sum(1 for v in neg if v > t)
        TN = len(neg) - FP
        re = TP / (TP + FN) if TP + FN else 0
        fp = FP / (FP + TN) if FP + TN else 0
        pr = TP / (TP + FP) if TP + FP else 0
        f1 = 2 * pr * re / (pr + re) if pr + re else 0
        return re, fp, pr, f1

    thresholds = sorted(set(round(v, 5) for v in pos + neg))
    band_best = {}
    for t in thresholds:
        re, fp, pr, f1 = stats_at(t)
        band = round(re * 20) / 20
        if band not in band_best or fp < band_best[band][0]:
            band_best[band] = (fp, t, f1, pr)
    print(f'    Pareto frontier (min FPR per recall band):')
    print(f'    {"recall":>7s}  {"min_FPR":>7s}  {"threshold":>12s}  {"prec":>5s}  {"F1":>5s}')
    for band in sorted(band_best, reverse=True):
        fp, t, f1, pr = band_best[band]
        print(f'    {band*100:>5.0f}%   {fp*100:>5.1f}%   {t:>11.4f}   {pr*100:>4.0f}%   {f1:.2f}')
    best_t, best_f1 = None, -1
    for t in thresholds:
        re, fp, pr, f1 = stats_at(t)
        if f1 > best_f1:
            best_t, best_f1, best_re, best_fp, best_pr = t, f1, re, fp, pr
    print(f'    BEST F1: t={best_t:.4f}  recall={best_re:.0%}  '
          f'FPR={best_fp:.0%}  prec={best_pr:.0%}  F1={best_f1:.3f}')


def _collate(b):
    keys = ('volume', 'target_slice_rgb', 'plane_idx', 'slice_idx_norm',
            'voxel_spacing', 'intensity_hist')
    return {k: torch.stack([s[k] for s in b]) for k in keys}


def _score_volume(model, path, in_channels, volume_size, n_slices, device):
    """Score one volume → list of per-slice anomaly values (axial only)."""
    ds = Vol2SliceDataset(
        scan_paths=[path],
        volume_size=volume_size, in_channels=in_channels,
        slices_per_volume=n_slices, slice_resize_hw=(256, 256),
        planes=('axial',),     # only axial for this eval
        shuffle=False, seed=0,
    )
    per_slice = []
    loader = DataLoader(ds, batch_size=1, collate_fn=_collate)
    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        with torch.no_grad():
            score = model.anomaly_score(
                batch['volume'], batch['target_slice_rgb'],
                batch['plane_idx'], batch['slice_idx_norm'],
                batch['voxel_spacing'], batch['intensity_hist'])
        per_slice.append(float(score.item()))
    return per_slice


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')

    # ---- Load model ----
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    args = ck.get('args', {})
    vol_size = tuple(args.get('volume_size', [144, 192, 192]))
    in_channels = args.get('in_channels', 1)
    print(f'[init] checkpoint: {ck.get("epoch")} epochs, vol_size={vol_size}')

    teacher = V8FrozenTeacher.from_unet_checkpoint(
        str(V8_PT), encoder_name='tu-convnext_tiny.fb_in22k_ft_in1k',
        in_channels=3, image_size=384, device=device)
    model = Vol2SliceModel(
        v8_teacher=teacher, volume_size=vol_size, in_chans=in_channels,
        patch_size=args.get('patch_size', 16),
        encoder_dim=args.get('encoder_dim', 384),
        encoder_depth=args.get('encoder_depth', 12),
        encoder_heads=args.get('encoder_heads', 6),
        predictor_dim=args.get('predictor_dim', 192),
        predictor_depth=args.get('predictor_depth', 6),
    ).to(device)
    model.encoder.load_state_dict(ck['encoder_state_dict'])
    model.predictor.load_state_dict(ck['predictor_state_dict'])
    model.conditioning.load_state_dict(ck['conditioning_state_dict'])
    model.eval()

    # ---- Sample volumes ----
    rng = random.Random(0)
    healthy_pool = sorted(glob.glob(HEALTHY_GLOB))
    tumor_pool = []
    for d in sorted(BRATS_ROOT.iterdir()):
        if d.is_dir():
            flair = list(d.glob('*flair.nii.gz'))
            if flair:
                tumor_pool.append(str(flair[0]))

    rng.shuffle(healthy_pool); rng.shuffle(tumor_pool)
    healthy_eval = healthy_pool[:N_HEALTHY]
    tumor_eval = tumor_pool[:N_TUMOR]
    print(f'[init] pool: {len(healthy_pool)} healthy, {len(tumor_pool)} tumor')
    print(f'[init] eval: {len(healthy_eval)} healthy + {len(tumor_eval)} tumor')

    # ============================================================
    # PART A: per-volume anomaly = mean over N axial slices
    # ============================================================
    print('\n' + '=' * 70)
    print('PART A: per-volume score = mean of 8 axial-slice anomaly scores')
    print('=' * 70)

    rows_a = []
    t0 = time.perf_counter()
    last_report = t0
    n_done = 0
    for label, paths in [('no_tumor', healthy_eval), ('tumor', tumor_eval)]:
        for p in paths:
            try:
                slices = _score_volume(model, p, in_channels, vol_size,
                                         N_SLICES_PART_A, device)
            except Exception as exc:
                print(f'  [skip] {Path(p).stem[:40]}: {exc}')
                continue
            rows_a.append({
                'gt': label, 'path': p, 'scan_id': Path(p).stem,
                'mean_score': float(np.mean(slices)),
                'max_score': float(np.max(slices)),
                'p95_score': float(np.percentile(slices, 95)),
                'n_slices': len(slices),
            })
            n_done += 1
            if time.perf_counter() - last_report > 20:
                last_report = time.perf_counter()
                rate = n_done / (time.perf_counter() - t0)
                eta = (len(healthy_eval) + len(tumor_eval) - n_done) / max(rate, 1e-6)
                print(f'  [{n_done}/{len(healthy_eval)+len(tumor_eval)}]  '
                      f'rate={rate:.2f}/s  eta={eta:.0f}s')
    print(f'  Part A done in {(time.perf_counter()-t0)/60:.1f} min')

    # Honest caveat
    print('\n  NOTE: the radiata-ai healthy volumes used here were in the')
    print('  training set. Healthy scores are an optimistic lower bound;')
    print('  a proper held-out healthy test needs additional fresh data.')

    print('\n  Part A — score = MEAN over slices:')
    _pareto_report(rows_a, 'mean_score')

    # ============================================================
    # PART B: aggregation sweep on a smaller cohort with denser slices
    # ============================================================
    print('\n' + '=' * 70)
    print(f'PART B: aggregation sweep — score every {N_SLICES_PART_B} axial slices,')
    print(f'        try mean / max / p95 / top-k mean over each volume.')
    print('=' * 70)

    rng2 = random.Random(1)
    h_b = rng2.sample(healthy_eval, N_HEALTHY_PART_B)
    t_b = rng2.sample(tumor_eval, N_TUMOR_PART_B)
    rows_b = []
    t0 = time.perf_counter()
    n_done = 0
    for label, paths in [('no_tumor', h_b), ('tumor', t_b)]:
        for p in paths:
            try:
                slices = _score_volume(model, p, in_channels, vol_size,
                                         N_SLICES_PART_B, device)
            except Exception as exc:
                print(f'  [skip] {Path(p).stem[:40]}: {exc}')
                continue
            arr = np.asarray(slices)
            top4 = float(np.mean(np.sort(arr)[-4:]))   # top-4 mean (k=4)
            rows_b.append({
                'gt': label, 'path': p, 'scan_id': Path(p).stem,
                'mean_score': float(arr.mean()),
                'max_score': float(arr.max()),
                'p95_score': float(np.percentile(arr, 95)),
                'p99_score': float(np.percentile(arr, 99)),
                'topk_mean_score': top4,
                'n_slices': len(slices),
            })
            n_done += 1
    print(f'  Part B done in {(time.perf_counter()-t0)/60:.1f} min')

    print('\n  Aggregation comparison:')
    for agg_key in ('mean_score', 'max_score', 'p95_score', 'p99_score',
                    'topk_mean_score'):
        pos = [r[agg_key] for r in rows_b if r['gt'] == 'tumor']
        neg = [r[agg_key] for r in rows_b if r['gt'] == 'no_tumor']
        auc = _auc(pos, neg)
        print(f'\n  --- aggregation: {agg_key} ---')
        print(f'    AUC = {auc:.4f}')
        _pareto_report(rows_b, agg_key)

    # ---- CSV dump (Part A + Part B combined) ----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        all_rows = [{**r, 'part': 'A', 'p99_score': None,
                     'topk_mean_score': None} for r in rows_a]
        all_rows += [{**r, 'part': 'B'} for r in rows_b]
        fieldnames = ['part', 'gt', 'scan_id', 'mean_score', 'max_score',
                       'p95_score', 'p99_score', 'topk_mean_score',
                       'n_slices', 'path']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f'\n[csv] {OUT_CSV}')


if __name__ == '__main__':
    main()
