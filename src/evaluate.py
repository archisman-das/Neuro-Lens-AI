import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from src.data import get_datasets, prepare_dataset
from src.models import get_model
from src.utils import compute_metrics, save_metrics_json


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained brain tumor detection models')
    parser.add_argument('--model', choices=['cnn', 'transfer', 'vit'], default='cnn')
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', default='artifacts')
    return parser.parse_args()


def main():
    args = parse_args()
    train_ds, val_ds, test_ds = get_datasets(args.dataset, batch_size=args.batch_size)
    if test_ds is None:
        print('No test split found in dataset. Evaluation requires dataset/test or a separate evaluation dataset.')
        return
    test_ds = prepare_dataset(test_ds)
    model = get_model(args.model, transfer_weights=None)
    model.load_weights(args.weights)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
    )
    result = model.evaluate(test_ds, verbose=1)
    print('Raw evaluation results:', result)

    metrics = compute_metrics(model, test_ds)
    os.makedirs(args.output, exist_ok=True)
    metrics_path = os.path.join(args.output, f'{args.model}_evaluation_metrics.json')
    save_metrics_json(metrics, metrics_path)
    print(f'Evaluation metrics saved to {metrics_path}')
    print('Classification report:')
    print(metrics['classification_report'])


if __name__ == '__main__':
    main()
