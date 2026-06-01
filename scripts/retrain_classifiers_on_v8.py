"""Retrain CNN / Transfer / ViT classifiers on dataset_v8.

Why this exists (the brutal version): the live classifiers were trained
on Kaggle 4-class only and hit 0%-67% recall on OOD tumors (see
scripts/eval_ood_classifiers_brutal.py output). dataset_v8 is the same
distribution the v8 *segmenter* trains on (BraTS T1c + LGG + Figshare +
Kaggle 4-class neg), so retraining the classifiers there closes the gap
between segmenter generalisation and classifier generalisation.

What's in / not in:
  - IN (train/val/test): dataset_v8/{train,val,test} ONLY.
  - NOT IN: samples/ood/* — those are held out for the final production
    accuracy number, on the user's explicit instruction.

Labels derived from masks: a sample is tumor iff its mask has >= 50
tumor pixels (matches MIN_TUMOR_AREA used everywhere else in the repo).

Outputs to real_eval_v8_retrained/<model>/{best_weights.pt, best_weights.onnx}.
The dashboard's _classifier_dir() resolver checks real_eval_v8_retrained/
first when it exists, so a successful run flips production automatically.

Run:
  python scripts/retrain_classifiers_on_v8.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classifier_torch import get_classifier  # noqa: E402

MIN_TUMOR_AREA = 50
IMAGE_SIZE = 224
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MaskDerivedClassificationDataset(Dataset):
    """Reads dataset_v8/<split>/images/*.png + masks/*.png.

    Label: 1 if mask has >= MIN_TUMOR_AREA tumor pixels, else 0.
    Caches the per-image label once at __init__ so the per-getitem hot
    path only reads the image.
    """

    def __init__(self, split_dir: Path, image_size: int = IMAGE_SIZE,
                 normalize_imagenet: bool = False, train: bool = True):
        self.image_size = image_size
        self.normalize_imagenet = normalize_imagenet
        self.train = train
        self.images_dir = Path(split_dir) / 'images'
        self.masks_dir = Path(split_dir) / 'masks'
        if not self.images_dir.exists():
            raise FileNotFoundError(f'no images dir at {self.images_dir}')
        if not self.masks_dir.exists():
            raise FileNotFoundError(f'no masks dir at {self.masks_dir}')
        # Pre-compute labels (cheap: a single np.sum per mask).
        entries = []
        n_tum = n_neg = 0
        for img_path in sorted(self.images_dir.glob('*.png')):
            mask_path = self.masks_dir / img_path.name
            if not mask_path.exists():
                continue
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # noqa: F811
            label = 1.0 if int((m > 127).sum()) >= MIN_TUMOR_AREA else 0.0
            entries.append((img_path, label))
            if label == 1.0: n_tum += 1
            else: n_neg += 1
        self.entries = entries
        self.n_tumor = n_tum
        self.n_no_tumor = n_neg
        print(f'  [dataset] {split_dir.name:5s}: total={len(entries):5d}  '
              f'tumor={n_tum:5d}  no_tumor={n_neg:5d}  '
              f'class_balance={n_tum/max(len(entries),1):.1%} positive')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f'failed to read {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        if self.train and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
        img = img.astype(np.float32) / 255.0
        if self.normalize_imagenet:
            img = (img - IM_MEAN) / IM_STD
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device, threshold: float = 0.5) -> dict:
    model.eval()
    y_true, y_pred_prob, y_pred_bin = [], [], []
    bce_total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).squeeze(-1)
            probs = torch.sigmoid(logits)
            bce_total += F.binary_cross_entropy_with_logits(logits, y, reduction='sum').item()
            y_true.extend(y.cpu().numpy().tolist())
            y_pred_prob.extend(probs.cpu().numpy().tolist())
            y_pred_bin.extend((probs >= threshold).float().cpu().numpy().tolist())
            n += y.shape[0]
    y_true = np.asarray(y_true); y_pred_bin = np.asarray(y_pred_bin); y_pred_prob = np.asarray(y_pred_prob)
    tp = int(((y_true==1)&(y_pred_bin==1)).sum()); fp = int(((y_true==0)&(y_pred_bin==1)).sum())
    fn = int(((y_true==1)&(y_pred_bin==0)).sum()); tn = int(((y_true==0)&(y_pred_bin==0)).sum())
    accuracy = (tp+tn)/max(n,1); precision = tp/max(tp+fp,1)
    recall = tp/max(tp+fn,1); f1 = 2*precision*recall/max(precision+recall,1e-9)
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(y_true, y_pred_prob)) if len(set(y_true)) > 1 else float('nan')
    except Exception:
        roc_auc = float('nan')
    return {'n': n, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'roc_auc': roc_auc,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'bce_loss_mean': bce_total/max(n,1)}


def export_onnx(model: nn.Module, save_path: Path, device):
    """Export to ONNX. Redirects the exporter's emoji-heavy stdout/stderr
    into an in-memory StringIO so it never hits the parent process's
    cp1252 console (which would crash on the success checkmark)."""
    import io
    import contextlib
    model.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    buf = io.StringIO()
    ok = False
    err: Exception | None = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            torch.onnx.export(
                model, dummy, str(save_path),
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=17,
            )
        ok = True
    except Exception as exc:
        err = exc
    if ok:
        print(f'        -> exported ONNX: {save_path} ({save_path.stat().st_size/1e6:.1f} MB)')
    else:
        print(f'        ONNX export failed (continuing): {type(err).__name__}: {err}')


def train_one(model_name: str, args) -> dict:
    print(f'\n========== training {model_name} on dataset_v8 ==========', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[{model_name}] device={device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''), flush=True)

    normalize = (model_name != 'cnn')
    train_ds = MaskDerivedClassificationDataset(Path(args.dataset) / 'train',
                                                  normalize_imagenet=normalize, train=True)
    val_ds = MaskDerivedClassificationDataset(Path(args.dataset) / 'val',
                                                normalize_imagenet=normalize, train=False)
    test_ds = MaskDerivedClassificationDataset(Path(args.dataset) / 'test',
                                                 normalize_imagenet=normalize, train=False)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                   pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = get_classifier(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{model_name}] trainable params: {n_params:,}', flush=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Compute pos_weight to balance the BCE loss against class imbalance.
    # pos_weight = n_neg / n_pos. When pos_weight < 1 we downweight the
    # majority (positives, after OpenNeuro augmentation); when > 1 we
    # upweight the minority. Without this the previous round learned a
    # positive bias and false-alarmed on all OOD healthy brains.
    pw = float(train_ds.n_no_tumor) / max(train_ds.n_tumor, 1)
    pw_tensor = torch.tensor([pw], device=device, dtype=torch.float32)
    print(f'[{model_name}] pos_weight = n_neg/n_pos = '
          f'{train_ds.n_no_tumor}/{train_ds.n_tumor} = {pw:.4f}', flush=True)

    out_dir = ROOT / args.output / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / 'best_weights.pt'
    onnx_path = out_dir / 'best_weights.onnx'

    best_val_acc = -1.0
    epochs_without_improve = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        if hasattr(model, 'backbone'):
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        running_loss = 0.0; running_correct = 0; running_n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw_tensor)
            if device.type == 'cuda':
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            running_loss += float(loss) * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            running_correct += int((preds == y).sum().item())
            running_n += x.size(0)
        train_loss = running_loss/max(running_n,1); train_acc = running_correct/max(running_n,1)
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics['accuracy']; val_loss = val_metrics['bce_loss_mean']
        elapsed = time.time() - t0
        print(f'[{model_name}][ep {epoch+1:02d}/{args.epochs}] '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f}  '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}  '
              f'val_recall={val_metrics["recall"]:.4f}  ({elapsed:.1f}s)', flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save({'state_dict': model.state_dict(), 'model_name': model_name,
                         'val_metrics': val_metrics, 'epoch': epoch+1,
                         'normalize_imagenet': normalize}, best_path)
            print(f'        -> new best val_acc={best_val_acc:.4f}; saved {best_path}', flush=True)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f'[{model_name}] early stopping at ep {epoch+1}', flush=True)
                break

    # Final: load best, eval test, export ONNX
    ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    test_metrics = evaluate(model, test_loader, device)
    print(f'[{model_name}] TEST: acc={test_metrics["accuracy"]:.4f} '
          f'recall={test_metrics["recall"]:.4f} precision={test_metrics["precision"]:.4f} '
          f'f1={test_metrics["f1"]:.4f} roc_auc={test_metrics["roc_auc"]:.4f}', flush=True)
    final = {'val': evaluate(model, val_loader, device), 'test': test_metrics}
    (ROOT / args.output / f'{model_name}_evaluation_metrics.json').write_text(
        json.dumps(final, indent=2), encoding='utf-8')
    export_onnx(model, onnx_path, device)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset_v8')
    parser.add_argument('--output', default='real_eval_v8_retrained')
    parser.add_argument('--models', nargs='+', default=['cnn', 'transfer', 'vit'],
                        choices=['cnn', 'transfer', 'vit'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f'[init] cuda={torch.cuda.is_available()} dataset={args.dataset} output={args.output}')
    print(f'[init] models={args.models} epochs={args.epochs} batch={args.batch_size}')
    print(f'[init] NOT touching samples/ood/* — those are held out for production accuracy.')
    results = {}
    t_total = time.time()
    for m in args.models:
        results[m] = train_one(m, args)
    print(f'\n[done] all classifiers trained in {(time.time()-t_total)/60:.1f} min')
    print('\nSUMMARY (test split, dataset_v8):')
    for m, r in results.items():
        t = r['test']
        print(f'  {m:10s}  acc={t["accuracy"]:.4f}  recall={t["recall"]:.4f}  '
              f'precision={t["precision"]:.4f}  f1={t["f1"]:.4f}  auc={t["roc_auc"]:.4f}')


if __name__ == '__main__':
    main()
