import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
import tensorflow as tf
from src.data import get_datasets, prepare_dataset, get_augmentation_layer
from src.models import get_model
from src.utils import save_history, plot_training_history
try:
    from src.config_loader import set_yaml_defaults
except Exception:  # pragma: no cover - pyyaml optional
    set_yaml_defaults = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train brain tumor detection models')
    parser.add_argument('--model', choices=['cnn', 'transfer', 'vit'], default='cnn')
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--validation_split', type=float, default=0.15)
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--fine_tune_transfer', action='store_true', help='Unfreeze the upper layers of the transfer backbone.')
    parser.add_argument('--transfer_fine_tune_at', type=int, default=140, help='Layer index where transfer fine-tuning starts.')
    parser.add_argument('--augment', action='store_true', help='Apply random flip/rotation/zoom/contrast augmentation on the train split.')
    parser.add_argument('--config', default=None, help='Optional path to config.yaml to use for default values.')
    # YAML defaults: read the [training] section of config.yaml and apply as
    # parser defaults. CLI flags still win. Mapping below is explicit since the
    # YAML keys don't all match argparse attribute names.
    pre_args, _ = parser.parse_known_args()
    if set_yaml_defaults is not None:
        try:
            set_yaml_defaults(
                parser,
                'training',
                mapping={
                    'epochs': 'epochs',
                    'batch_size': 'batch_size',
                    'learning_rate': 'learning_rate',
                },
                path=pre_args.config,
            )
        except FileNotFoundError:
            pass
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    train_ds, val_ds, test_ds = get_datasets(
        args.dataset,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )
    # Optional train-time augmentation. The aug layer also rescales to [0,1] so
    # we keep the in-model Rescaling unchanged: aug layer outputs float [0,1],
    # the in-model Rescaling(1/255) gets a near-no-op since inputs are already
    # small floats. Pass through the float tensor; for transfer/vit the
    # ResNet50 preprocess_input remains valid (it accepts floats).
    if args.augment:
        aug = get_augmentation_layer(image_size=(224, 224))
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True) * 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
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
