"""Per-volume z-score normalized eval — fix (B) from the held-out failure.

Hypothesis: the raw absolute anomaly score for a volume is a sum of
two unrelated terms:
  1. Preprocessing-pipeline shift (constant across all slices of that
     volume)
  2. Per-slice pathology signal (localized to slices that contain tumor)

By z-scoring each volume's per-slice anomaly distribution against its
own mean and std, we cancel out (1) and surface (2). Tumor volumes
should have a few extreme-z slices (where the tumor is); healthy
volumes should have all slices near z=0 regardless of the absolute
offset.

Volume score = high quantile (max / p95 / top-k mean) of the z-scored
per-slice anomalies.

We use DENSER sampling (20 slices, central 50%) so the z-score has a
meaningful distribution to normalize against.
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


CKPT = ROOT / 'v9b_artifacts' / 'v9c_crossjepa_method1' / 'last.pt'
V8_PT = ROOT / 'model' / 'best_micro.pt'
IXI_GLOB = 'd:/datasets/ixi_heldout/ixi_preprocessed/IXI*-T1.nii.gz'
RADIATA_GLOB = ('c:/Users/anish/.cache/huggingface/hub/'
                 'datasets--radiata-ai--brain-structure/snapshots/'
                 'aea73e2a9734955407eb8bcf2325b479a5f39144/'
                 '*/sub-*/ses-*/anat/*.nii.gz')
BRATS_ROOT = Path('d:/datasets/brats2021')
OUT_CSV = ROOT / 'samples' / 'ood' / 'eval_method1_zscore_results.csv'

N_TUMOR = 50
N_HEALTHY_IXI = 50           # held-out
N_HEALTHY_RADIATA = 30        # train-set, for reference (should still score low even with z-score)
N_SLICES = 20                 # need a decent distribution to z-score against


def _collate(b):
    keys = ('volume', 'target_slice_rgb', 'plane_idx', 'slice_idx_norm',
            'voxel_spacing', 'intensity_hist')
    return {k: torch.stack([s[k] for s in b]) for k in keys}


def _score_volume(model, path, in_channels, volume_size, n_slices, device):
    ds = Vol2SliceDataset(
        scan_paths=[path], volume_size=volume_size, in_channels=in_channels,
        slices_per_volume=n_slices, slice_resize_hw=(256, 256),
        planes=('axial',), shuffle=False, seed=0,
    )
    per_slice = []
    for batch in DataLoader(ds, batch_size=1, collate_fn=_collate):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        with torch.no_grad():
            s = model.anomaly_score(
                batch['volume'], batch['target_slice_rgb'],
                batch['plane_idx'], batch['slice_idx_norm'],
                batch['voxel_spacing'], batch['intensity_hist'])
        per_slice.append(float(s.item()))
    return np.asarray(per_slice)


def _z_aggregate(slices: np.ndarray) -> dict:
    """Return the per-volume aggregate scores AFTER z-score normalization
    within the volume."""
    raw_max = float(slices.max())
    raw_mean = float(slices.mean())
    raw_p95 = float(np.percentile(slices, 95))
    # z-score within the volume
    mu = slices.mean()
    sd = slices.std()
    if sd < 1e-9:
        z = np.zeros_like(slices)
    else:
        z = (slices - mu) / sd
    return {
        'raw_mean': raw_mean, 'raw_max': raw_max, 'raw_p95': raw_p95,
        'z_max': float(z.max()),
        'z_p95': float(np.percentile(z, 95)),
        'z_topk4_mean': float(np.mean(np.sort(z)[-4:])),
        'z_range': float(z.max() - z.min()),
        'volume_mean': float(mu),
        'volume_std': float(sd),
    }


def _auc(pos, neg):
    if not pos or not neg: return float('nan')
    wins = ties = 0
    for sp in pos:
        for sn in neg:
            if sp > sn: wins += 1
            elif sp == sn: ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    args = ck.get('args', {})
    vol_size = tuple(args.get('volume_size', [144, 192, 192]))
    in_channels = args.get('in_channels', 1)

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

    rng = random.Random(0)

    # Pools
    ixi_paths = sorted(glob.glob(IXI_GLOB))[:N_HEALTHY_IXI]
    radiata_pool = sorted(glob.glob(RADIATA_GLOB))
    rng.shuffle(radiata_pool)
    radiata_paths = radiata_pool[:N_HEALTHY_RADIATA]
    tumor_pool = []
    for d in sorted(BRATS_ROOT.iterdir()):
        if d.is_dir():
            flair = list(d.glob('*flair.nii.gz'))
            if flair: tumor_pool.append(str(flair[0]))
    rng.shuffle(tumor_pool)
    tumor_paths = tumor_pool[:N_TUMOR]

    print(f'[init] IXI held-out healthy: {len(ixi_paths)}')
    print(f'[init] radiata-ai train-set healthy (reference): {len(radiata_paths)}')
    print(f'[init] BraTS tumor: {len(tumor_paths)}')

    rows = []
    t0 = time.perf_counter()
    n_done = 0
    for group, label, paths in [
        ('ixi_heldout', 'no_tumor', ixi_paths),
        ('radiata_train', 'no_tumor', radiata_paths),
        ('brats_tumor', 'tumor', tumor_paths),
    ]:
        print(f'\n[scoring] {group} ({label}): {len(paths)} volumes...')
        for p in paths:
            try:
                slices = _score_volume(model, p, in_channels, vol_size,
                                         N_SLICES, device)
                agg = _z_aggregate(slices)
            except Exception as exc:
                print(f'  [skip] {Path(p).stem[:40]}: {exc}')
                continue
            row = {'group': group, 'gt': label, 'scan_id': Path(p).stem, **agg}
            rows.append(row)
            n_done += 1
        print(f'  done {len(paths)} ({n_done} cumulative, '
              f'{time.perf_counter()-t0:.0f}s elapsed)')
    print(f'\n[done] total {len(rows)} volumes in {(time.perf_counter()-t0)/60:.1f} min')

    # Report — multiple aggregations
    print('\n' + '=' * 60)
    print('  AUC across aggregations  (IXI held-out vs BraTS tumor)')
    print('=' * 60)
    ixi_rows = [r for r in rows if r['group'] == 'ixi_heldout']
    brats_rows = [r for r in rows if r['group'] == 'brats_tumor']
    radiata_rows = [r for r in rows if r['group'] == 'radiata_train']

    for key in ('raw_mean', 'raw_max', 'raw_p95',
                'z_max', 'z_p95', 'z_topk4_mean'):
        pos = [r[key] for r in brats_rows]
        neg = [r[key] for r in ixi_rows]
        auc = _auc(pos, neg)
        print(f'\n  {key}:  AUC = {auc:.4f}  (tumor n={len(pos)}, healthy n={len(neg)})')
        print(f'    tumor:    mean={np.mean(pos):.4f}  '
              f'[min,med,max]=[{min(pos):.3f},{np.median(pos):.3f},{max(pos):.3f}]')
        print(f'    healthy:  mean={np.mean(neg):.4f}  '
              f'[min,med,max]=[{min(neg):.3f},{np.median(neg):.3f},{max(neg):.3f}]')
        # Best-F1 sweep
        all_vals = sorted(set(pos + neg))
        best_t, best_f1 = None, -1
        for t in all_vals:
            TP = sum(1 for v in pos if v > t)
            FN = len(pos) - TP
            FP = sum(1 for v in neg if v > t)
            TN = len(neg) - FP
            re = TP/(TP+FN) if TP+FN else 0
            fp = FP/(FP+TN) if FP+TN else 0
            pr = TP/(TP+FP) if TP+FP else 0
            f1 = 2*pr*re/(pr+re) if pr+re else 0
            if f1 > best_f1:
                best_t, best_f1, best_re, best_fp, best_pr = t, f1, re, fp, pr
        if best_t is not None:
            print(f'    BEST F1: t={best_t:.4f}  recall={best_re:.0%}  '
                  f'FPR={best_fp:.0%}  prec={best_pr:.0%}  F1={best_f1:.3f}')

    # Also show: does z-score keep the radiata-vs-brats separation strong?
    print('\n  z_max sanity check on radiata_train (training-set healthy):')
    if radiata_rows:
        r_zmax = [r['z_max'] for r in radiata_rows]
        print(f'    radiata train  mean={np.mean(r_zmax):.4f}  '
              f'[min,med,max]=[{min(r_zmax):.3f},{np.median(r_zmax):.3f},{max(r_zmax):.3f}]')
        ixi_zmax = [r['z_max'] for r in ixi_rows]
        brats_zmax = [r['z_max'] for r in brats_rows]
        print(f'    ixi held-out   mean={np.mean(ixi_zmax):.4f}')
        print(f'    brats tumor    mean={np.mean(brats_zmax):.4f}')
        print('    Expect: radiata~ixi (both healthy) << brats (tumor)')

    # CSV dump
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\n[csv] {OUT_CSV}')


if __name__ == '__main__':
    main()
