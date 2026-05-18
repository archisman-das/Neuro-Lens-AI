import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
import tensorflow as tf
from src.data import get_datasets, prepare_dataset
from src.models import get_model
from src.utils import save_history, plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description='Train brain tumor detection models')
    parser.add_argument('--model', choices=['cnn', 'transfer', 'vit'], default='cnn')
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--fine_tune_transfer', action='store_true', help='Unfreeze the upper layers of the transfer backbone.')
    parser.add_argument('--transfer_fine_tune_at', type=int, default=140, help='Layer index where transfer fine-tuning starts.')
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    train_ds, val_ds, test_ds = get_datasets(
        args.dataset,
        batch_size=args.batch_size,
        validation_split=0.15,
    )
    train_ds = prepare_dataset(train_ds)
    val_ds = prepare_dataset(val_ds)

    model = get_model(
        model_name,
        fine_tune_transfer=args.fine_tune_transfer,
        transfer_fine_tune_at=args.transfer_fine_tune_at,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(args.output, model_name)
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, 'best_weights.weights.h5')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    history_path = os.path.join(model_dir, f'history_{timestamp}.npz')
    save_history(history, history_path)
    plot_training_history(history, model_dir)
    print(f'Model training complete. Weights and history saved to {model_dir}')


if __name__ == '__main__':
    main()
