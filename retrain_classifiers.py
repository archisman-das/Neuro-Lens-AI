"""Retrain the CNN / Transfer / ViT classifiers locally.

Needed because:
  - The classifier .h5 files in real_eval_current/ and real_eval_fixed/ are
    134-byte Git LFS pointer files (the zip download from GitHub never fetched
    the binaries), and
  - The one real .h5 the upstream repo still hosts (real_eval_current/vit) has
    a topology that doesn't match the current src/models.py build_vit_classifier
    (saved with ResNet50 flattened; current code nests it). Partial-loading
    keeps every layer at its randomly-initialised / ImageNet-default values.

This script trains all three classifiers on dataset_real/ at the same
hyperparameters used by src/train.py, writing weights into
real_eval_current/<model>/best_weights.weights.h5 so the dashboard finds them
without any further changes.

NOTE on speed: TF 2.21 has no native-Windows GPU. Expect CPU training to take
roughly:
  - cnn       : 20-40 min (small model, ~0.4M params)
  - transfer  : 40-90 min (ResNet50 backbone frozen, only head trains)
  - vit       : 40-90 min (4 transformer blocks)

Reduce --epochs if you just want a quick demo set; default 10 mirrors the
original training script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

import tensorflow as tf  # noqa: E402
from src.data import get_datasets, prepare_dataset  # noqa: E402
from src.models import get_model  # noqa: E402
from src.utils import compute_metrics, save_metrics_json  # noqa: E402


def train_one(model_name: str, dataset_dir: str, epochs: int, batch_size: int, lr: float, output_root: Path):
    print(f'\n========== training {model_name} ==========', flush=True)
    train_ds, val_ds, test_ds = get_datasets(dataset_dir, batch_size=batch_size, validation_split=0.15)
    train_ds = prepare_dataset(train_ds)
    val_ds = prepare_dataset(val_ds)
    test_ds = prepare_dataset(test_ds) if test_ds is not None else val_ds

    model = get_model(model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
    )

    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / 'best_weights.weights.h5'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Persist evaluation metrics next to the weights so the dashboard's
    # /metrics endpoint surfaces them. compute_metrics expects an unbatched
    # iterable of (image, label) batches; the cached val_ds works.
    model.load_weights(str(weights_path))
    metrics = compute_metrics(model, test_ds)
    metrics_path = output_root / f'{model_name}_evaluation_metrics.json'
    save_metrics_json(metrics, metrics_path)
    print(f'[{model_name}] saved weights -> {weights_path}')
    print(f'[{model_name}] saved metrics -> {metrics_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset_real')
    parser.add_argument('--output', default='real_eval_current',
                        help='Output root (matches dashboard.py weight search order).')
    parser.add_argument('--models', nargs='+', default=['cnn', 'transfer', 'vit'],
                        choices=['cnn', 'transfer', 'vit'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    output_root = _REPO_ROOT / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        train_one(model_name, args.dataset, args.epochs, args.batch_size, args.learning_rate, output_root)

    print('\n[done] All requested classifiers retrained.')


if __name__ == '__main__':
    main()
