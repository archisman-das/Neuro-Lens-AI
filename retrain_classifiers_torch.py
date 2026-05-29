"""Retrain the 3 classifiers in PyTorch on GPU.

Replaces retrain_classifiers.py (TF/CPU, ~7 hours total) with a PyTorch
implementation that runs on the RTX 4060 in ~15 min total.

Outputs land in real_eval_current/<model>/best_weights.pt (state_dict only)
plus real_eval_current/<model>_evaluation_metrics.json. The dashboard auto-
detects PyTorch checkpoints and switches its loader to the torch path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.classifier_torch import get_classifier  # noqa: E402


class TumorClassificationDataset(Dataset):
    """Loads <split_dir>/tumor/*.jpg + <split_dir>/no_tumor/*.jpg as a binary task.

    Returns (image, label) where image is float[3,224,224] in [0,1] and label
    is float[] (1 = tumor, 0 = no_tumor). Per-channel ImageNet norm is applied
    INSIDE the model wrapper for the transfer/ViT paths (in the live dashboard);
    the bare CNN expects [0,1] without ImageNet normalization to match the
    original src/models.py:build_cnn_baseline behaviour.
    """

    def __init__(self, root: Path, image_size: int = 224, normalize_imagenet: bool = False,
                 train: bool = True):
        import cv2
        self.cv2 = cv2
        self.image_size = image_size
        self.normalize_imagenet = normalize_imagenet
        self.train = train
        root = Path(root)
        tumor = sorted(list((root / 'tumor').glob('*.jpg')) + list((root / 'tumor').glob('*.png')))
        no_tumor = sorted(list((root / 'no_tumor').glob('*.jpg')) + list((root / 'no_tumor').glob('*.png')))
        self.entries = [(p, 1.0) for p in tumor] + [(p, 0.0) for p in no_tumor]
        if not self.entries:
            raise FileNotFoundError(f'No tumor/no_tumor images found under {root}')
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        img = self.cv2.imread(str(path))
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = self.cv2.resize(img, (self.image_size, self.image_size))
        if self.train and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
        img = img.astype(np.float32) / 255.0
        if self.normalize_imagenet:
            img = (img - self.imagenet_mean) / self.imagenet_std
        img = img.transpose(2, 0, 1)  # CHW
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device, threshold: float = 0.5) -> dict:
    model.eval()
    y_true, y_pred_prob, y_pred_bin = [], [], []
    bce_total = 0.0
    n = 0
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

    y_true = np.asarray(y_true)
    y_pred_bin = np.asarray(y_pred_bin)
    y_pred_prob = np.asarray(y_pred_prob)
    tp = int(((y_true == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_bin == 0)).sum())
    tn = int(((y_true == 0) & (y_pred_bin == 0)).sum())
    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # ROC AUC via trapz on the sorted-probability ROC curve.
    try:
        from sklearn.metrics import roc_auc_score, classification_report
        roc_auc = float(roc_auc_score(y_true, y_pred_prob)) if len(set(y_true)) > 1 else float('nan')
        report = classification_report(y_true, y_pred_bin, output_dict=True, zero_division=0)
    except Exception:
        roc_auc = float('nan')
        report = None

    return {
        'n': n,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'bce_loss_mean': bce_total / max(n, 1),
        'classification_report': report,
    }


def train_one(model_name: str, args) -> dict:
    print(f'\n========== training {model_name} ==========', flush=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f'[{model_name}] device={device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''), flush=True)

    dataset_root = Path(args.dataset)
    normalize = (model_name != 'cnn')  # cnn was originally trained on [0,1] without ImageNet norm
    train_ds = TumorClassificationDataset(dataset_root / 'train', normalize_imagenet=normalize, train=True)
    val_ds = TumorClassificationDataset(dataset_root / 'val', normalize_imagenet=normalize, train=False)
    test_ds = TumorClassificationDataset(dataset_root / 'test', normalize_imagenet=normalize, train=False) \
        if (dataset_root / 'test').exists() else None
    print(f'[{model_name}] train={len(train_ds)} val={len(val_ds)}'
          + (f' test={len(test_ds)}' if test_ds else ''), flush=True)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common) if test_ds else None

    model = get_classifier(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{model_name}] trainable params: {n_params:,}', flush=True)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    out_root = _REPO_ROOT / args.output
    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / 'best_weights.pt'

    best_val_acc = -1.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    epochs_without_improve = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        # Keep BN frozen in eval mode for transfer/vit when backbone is frozen.
        if hasattr(model, 'backbone'):
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        running_loss = 0.0
        running_correct = 0
        running_n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += float(loss) * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            running_correct += int((preds == y).sum().item())
            running_n += x.size(0)

        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)

        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics['accuracy']
        val_loss = val_metrics['bce_loss_mean']
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f'[{model_name}][ep {epoch+1:02d}/{args.epochs}] '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f}  '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s)',
              flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save({
                'state_dict': model.state_dict(),
                'model_name': model_name,
                'val_metrics': val_metrics,
                'epoch': epoch + 1,
                'normalize_imagenet': normalize,
            }, best_path)
            print(f'        -> new best val_acc={best_val_acc:.4f}; saved {best_path}', flush=True)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f'[{model_name}] early stopping at ep {epoch+1}', flush=True)
                break

    # Load best, run final test
    ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    final = {'val': evaluate(model, val_loader, device)}
    if test_loader is not None:
        final['test'] = evaluate(model, test_loader, device)
    metrics_path = out_root / f'{model_name}_evaluation_metrics.json'
    metrics_path.write_text(json.dumps(final, indent=2), encoding='utf-8')
    print(f'[{model_name}] final metrics saved to {metrics_path}', flush=True)
    print(json.dumps({'val_acc': final["val"]["accuracy"],
                       'test_acc': final.get('test', {}).get('accuracy')}, indent=2))
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset_real')
    parser.add_argument('--output', default='real_eval_current')
    parser.add_argument('--models', nargs='+', default=['cnn', 'transfer', 'vit'],
                        choices=['cnn', 'transfer', 'vit'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = {}
    for m in args.models:
        results[m] = train_one(m, args)
    print('\n[done] All classifiers trained.')
    print(json.dumps({m: {'val_acc': r['val']['accuracy']} for m, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
