"""Evaluate a saved Attention U-Net checkpoint on val + test splits.

Re-runs the evaluation loop from train_segmentation_torch.py against the
checkpoint at segmentation_artifacts/attention_unet/best_model.pt and writes
evaluation_metrics.json. Useful after training crashes / is interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from src.segmentation_torch import AttentionUNet  # noqa: E402
from src.train_segmentation_torch import SegDataset, _evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='segmentation_artifacts/attention_unet/best_model.pt')
    parser.add_argument('--data_dir', default='dataset_real')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    print('Using device:', device)

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {}) or {}
    image_size = int(cfg.get('image_size', 256))
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch")} (image_size={image_size}, '
          f'base_filters={cfg.get("base_filters", 32)})')

    model = AttentionUNet(
        in_channels=3,
        base_filters=int(cfg.get('base_filters', 32)),
        dropout=float(cfg.get('dropout', 0.2)),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    data_dir = Path(args.data_dir)
    payload = {'source_checkpoint_epoch': ckpt.get('epoch'), 'config': cfg}
    for split in ['val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        ds = SegDataset(split_dir, image_size, augment=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = _evaluate(model, loader, device, threshold=args.threshold)
        payload[split] = metrics
        print(f'\n{split}: {len(ds)} samples')
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    out = Path(args.weights).parent / 'evaluation_metrics.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
