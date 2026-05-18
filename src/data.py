import tensorflow as tf
from pathlib import Path


def build_dataset(
    directory,
    image_size=(224, 224),
    batch_size=32,
    validation_split=None,
    subset=None,
    seed=123,
    shuffle=True,
):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        validation_split=validation_split,
        subset=subset,
        seed=seed,
    )


def get_datasets(
    root_dir,
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.15,
    seed=123,
):
    root = Path(root_dir)
    train_dir = root / 'train'
    val_dir = root / 'val'
    test_dir = root / 'test'

    if not root.exists():
        raise FileNotFoundError(f'Dataset root directory not found: {root_dir}')

    if val_dir.exists() and test_dir.exists():
        train_ds = build_dataset(train_dir, image_size=image_size, batch_size=batch_size, shuffle=True)
        val_ds = build_dataset(val_dir, image_size=image_size, batch_size=batch_size, shuffle=False)
        test_ds = build_dataset(test_dir, image_size=image_size, batch_size=batch_size, shuffle=False)
    elif train_dir.exists():
        train_ds = build_dataset(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            validation_split=validation_split,
            subset='training',
            seed=seed,
        )
        val_ds = build_dataset(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            validation_split=validation_split,
            subset='validation',
            seed=seed,
            shuffle=False,
        )
        test_ds = None
    else:
        raise FileNotFoundError(
            'Could not find expected train/val/test directories. Create `dataset/train` and optionally `dataset/val` and `dataset/test`.'
        )

    return train_ds, val_ds, test_ds


def get_augmentation_layer(image_size=(224, 224)):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Resizing(image_size[0], image_size[1]),
        ],
        name='data_augmentation',
    )


def prepare_dataset(dataset, cache=True, prefetch=True):
    if dataset is None:
        return None
    ds = dataset
    if cache:
        ds = ds.cache()
    if prefetch:
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
