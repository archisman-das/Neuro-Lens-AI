"""
Advanced Training Script with Robustness Analysis, Uncertainty Estimation, and Multiclass Classification
"""

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import os
from datetime import datetime

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from src.segmentation_models import build_unet, build_attention_unet, dice_loss, combined_loss
from src.robustness_analysis import RobustnessAnalyzer, UncertaintyEstimator, MulticlassSegmentationModel
from src.kfold_validation import SegmentationKFoldValidator


def train_with_robustness_analysis(config):
    """
    Train model and perform robustness analysis
    
    Args:
        config: Configuration dictionary
    """
    print("="*60)
    print("Training with Robustness Analysis")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config)
    
    # Build and train model
    model = build_attention_unet(
        input_shape=tuple(config.get('image_size', [224, 224])) + (3,),
        num_classes=1
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 1e-4)),
        loss=combined_loss(),
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 16),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
        ]
    )
    
    # Perform robustness analysis
    analyzer = RobustnessAnalyzer(model, input_shape=tuple(config.get('image_size', [224, 224])) + (3,))
    
    corruption_types = config.get('corruption_types', [
        'gaussian_noise', 'salt_pepper_noise', 'gaussian_blur',
        'brightness', 'contrast', 'rotation'
    ])
    
    corruption_levels = config.get('corruption_levels', [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    
    robustness_results = analyzer.evaluate_all_corruptions(
        X_test, y_test,
        corruption_types=corruption_types,
        corruption_levels=corruption_levels
    )
    
    # Save results
    save_dir = Path(config.get('save_dir', './robustness_training_results'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.save_results(robustness_results, str(save_dir))
    model.save(save_dir / 'robust_model.h5')
    
    print(f"\nRobustness Analysis Results:")
    for corruption_type, result in robustness_results.items():
        print(f"  {corruption_type}: Robustness Index = {result['robustness_index']:.3f}")
    
    return model, robustness_results


def train_with_uncertainty_estimation(config):
    """
    Train model and perform uncertainty estimation
    
    Args:
        config: Configuration dictionary
    """
    print("="*60)
    print("Training with Uncertainty Estimation")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config)
    
    # Build model with dropout for MC Dropout
    model = build_attention_unet(
        input_shape=tuple(config.get('image_size', [224, 224])) + (3,),
        num_classes=1,
        dropout_rate=config.get('dropout_rate', 0.3)  # Higher dropout for better uncertainty
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 1e-4)),
        loss=combined_loss(),
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 16),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
        ]
    )
    
    # Perform uncertainty estimation
    num_samples = config.get('num_samples', 50)
    estimator = UncertaintyEstimator(model, num_samples=num_samples)
    
    # MC Dropout predictions
    mean_pred, uncertainty, predictions_array = estimator.mc_dropout_predict(X_test)
    
    # Get confidence intervals
    lower_bound, upper_bound = estimator.get_confidence_intervals(
        predictions_array,
        confidence_level=config.get('confidence_level', 0.95)
    )
    
    # Save results
    save_dir = Path(config.get('save_dir', './uncertainty_training_results'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize uncertainty for first few samples
    for i in range(min(5, len(X_test))):
        estimator.visualize_uncertainty(
            X_test[i], mean_pred[i], uncertainty[i],
            save_path=save_dir / f'uncertainty_sample_{i}.png'
        )
    
    estimator.save_uncertainty_results(mean_pred, uncertainty, X_test[0], str(save_dir))
    model.save(save_dir / 'uncertainty_model.h5')
    
    # Print uncertainty statistics
    print(f"\nUncertainty Statistics:")
    print(f"  Mean uncertainty: {np.mean(uncertainty):.4f}")
    print(f"  Max uncertainty: {np.max(uncertainty):.4f}")
    print(f"  High uncertainty pixels (>0.5): {np.mean(uncertainty > 0.5):.2%}")
    print(f"  Medium uncertainty pixels (0.2-0.5): {np.mean((uncertainty > 0.2) & (uncertainty <= 0.5)):.2%}")
    print(f"  Low uncertainty pixels (<0.2): {np.mean(uncertainty <= 0.2):.2%}")
    
    return model, mean_pred, uncertainty


def train_multiclass_model(config):
    """
    Train multiclass segmentation model
    
    Args:
        config: Configuration dictionary
    """
    print("="*60)
    print("Training Multiclass Segmentation Model")
    print("="*60)
    
    # Load multiclass data
    X_train, y_train, X_val, y_val, X_test, y_test = load_multiclass_data(config)
    
    # Create multiclass model
    num_classes = config.get('num_classes', 4)
    model = MulticlassSegmentationModel(
        input_shape=tuple(config.get('image_size', [224, 224])) + (3,),
        num_classes=num_classes,
        base_filters=config.get('base_filters', 64),
        dropout_rate=config.get('dropout_rate', 0.2)
    )
    
    # Build and train
    model.build_model(use_attention=config.get('use_attention', True))
    model.compile_model(learning_rate=config.get('learning_rate', 1e-4))
    
    # Train
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 16)
    )
    
    # Evaluate
    metrics = model.evaluate_multiclass(X_test, y_test)
    
    # Save results
    save_dir = Path(config.get('save_dir', './multiclass_results'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(save_dir / 'multiclass_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Visualize predictions
    for i in range(min(5, len(X_test))):
        model.visualize_multiclass_prediction(
            X_test[i], y_test[i], model.predict(X_test[i:i+1])[0],
            save_path=save_dir / f'multiclass_prediction_{i}.png'
        )
    
    model.model.save(save_dir / 'multiclass_model.h5')
    
    # Print results
    print(f"\nMulticlass Evaluation Results:")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall: {class_metrics['recall']:.4f}")
        print(f"    F1 Score: {class_metrics['f1_score']:.4f}")
        print(f"    IoU: {class_metrics['iou']:.4f}")
        print(f"    Dice: {class_metrics['dice']:.4f}")
    
    print(f"\nMean Metrics:")
    for metric_name, value in metrics['mean_metrics'].items():
        print(f"  {metric_name}: {value:.4f}")
    
    return model, metrics


def _load_split_images_masks(split_dir, image_size):
    """Load (image, mask) pairs from a split directory.

    Expects either:
      <split_dir>/images/*.jpg|png and <split_dir>/masks/*.jpg|png  (paired by sorted order)
    or, if no masks dir is present, falls back to loading classification-style
    folders (tumor / no_tumor) and synthesising binary masks via Otsu thresholding
    on the intensity channel for tumor images (zero mask for no_tumor).
    """
    import cv2
    split_dir = Path(split_dir)
    images_dir = split_dir / 'images'
    masks_dir = split_dir / 'masks'

    def _read_image(path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f'Could not read image: {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        return img

    def _read_mask(path):
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f'Could not read mask: {path}')
        m = cv2.resize(m, image_size, interpolation=cv2.INTER_NEAREST)
        m = (m.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        return np.expand_dims(m, axis=-1)

    if images_dir.exists() and masks_dir.exists():
        image_paths = sorted([*images_dir.glob('*.png'), *images_dir.glob('*.jpg'), *images_dir.glob('*.jpeg')])
        mask_paths = sorted([*masks_dir.glob('*.png'), *masks_dir.glob('*.jpg'), *masks_dir.glob('*.jpeg')])
        if len(image_paths) != len(mask_paths):
            raise ValueError(f'Image/mask count mismatch in {split_dir}: {len(image_paths)} vs {len(mask_paths)}')
        X = np.stack([_read_image(p).astype(np.float32) for p in image_paths]) if image_paths else np.zeros((0, *image_size, 3), np.float32)
        y = np.stack([_read_mask(p) for p in mask_paths]) if mask_paths else np.zeros((0, *image_size, 1), np.float32)
        return X, y

    # Fallback: classification folders with synthesised masks via Otsu thresholding.
    tumor_dir = split_dir / 'tumor'
    no_tumor_dir = split_dir / 'no_tumor'
    if not tumor_dir.exists() and not no_tumor_dir.exists():
        raise FileNotFoundError(
            f'No images/masks/ or tumor/no_tumor/ subfolders found under {split_dir}.'
        )

    X_list = []
    y_list = []
    if tumor_dir.exists():
        for p in sorted([*tumor_dir.glob('*.png'), *tumor_dir.glob('*.jpg'), *tumor_dir.glob('*.jpeg')]):
            img = _read_image(p)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = (mask.astype(np.float32) / 255.0)
            X_list.append(img.astype(np.float32))
            y_list.append(np.expand_dims(mask, axis=-1))
    if no_tumor_dir.exists():
        for p in sorted([*no_tumor_dir.glob('*.png'), *no_tumor_dir.glob('*.jpg'), *no_tumor_dir.glob('*.jpeg')]):
            img = _read_image(p)
            X_list.append(img.astype(np.float32))
            y_list.append(np.zeros((*image_size, 1), np.float32))

    if not X_list:
        raise ValueError(f'No images found under {split_dir}.')
    return np.stack(X_list), np.stack(y_list)


def load_data(config):
    """Load binary segmentation data from the real dataset directory.

    Reads dataset_real/{train,val,test}/. If ground-truth masks are absent,
    pseudo-masks are synthesised via Otsu thresholding (see
    _load_split_images_masks). This was previously a random-noise placeholder.
    """
    data_dir = Path(config.get('data_dir', './dataset_real'))
    image_size = tuple(config.get('image_size', [224, 224]))

    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    if not train_dir.exists():
        raise FileNotFoundError(
            f'Training directory not found: {train_dir}. '
            'Run prepare_real_dataset.py or point --data_dir to a directory with train/, val/, test/.'
        )

    X_train, y_train = _load_split_images_masks(train_dir, image_size)
    if val_dir.exists():
        X_val, y_val = _load_split_images_masks(val_dir, image_size)
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=config.get('random_seed', 42)
        )
    if test_dir.exists():
        X_test, y_test = _load_split_images_masks(test_dir, image_size)
    else:
        X_test, y_test = X_val, y_val

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_multiclass_data(config):
    """Load multiclass segmentation data.

    Expects <data_dir>/<split>/images/*.png and <data_dir>/<split>/masks/*.png
    where mask pixel values encode the class id (0..num_classes-1). No real
    multiclass-segmentation data ships with this repo; this function will raise
    a clear error rather than silently train on noise (previous behaviour).
    """
    import cv2
    data_dir = Path(config.get('multiclass_data_dir', config.get('data_dir', './multiclass_dataset')))
    image_size = tuple(config.get('image_size', [224, 224]))
    num_classes = config.get('num_classes', 4)

    def _read_split(split):
        split_dir = data_dir / split
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'
        if not (images_dir.exists() and masks_dir.exists()):
            raise FileNotFoundError(
                f'Multiclass split missing images/ or masks/ at {split_dir}. '
                'Provide a dataset with per-pixel class id masks before training the multiclass model.'
            )
        image_paths = sorted([*images_dir.glob('*.png'), *images_dir.glob('*.jpg')])
        mask_paths = sorted([*masks_dir.glob('*.png'), *masks_dir.glob('*.jpg')])
        if len(image_paths) != len(mask_paths):
            raise ValueError(f'{split}: {len(image_paths)} images vs {len(mask_paths)} masks.')
        Xs, ys = [], []
        for ip, mp in zip(image_paths, mask_paths):
            img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size).astype(np.float32)
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, image_size, interpolation=cv2.INTER_NEAREST).astype(np.int32)
            m = np.clip(m, 0, num_classes - 1)
            Xs.append(img)
            ys.append(m)
        return np.stack(Xs), np.stack(ys)

    X_train, y_train = _read_split('train')
    X_val, y_val = _read_split('val') if (data_dir / 'val').exists() else (None, None)
    X_test, y_test = _read_split('test') if (data_dir / 'test').exists() else (None, None)

    if X_val is None:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=config.get('random_seed', 42)
        )
    if X_test is None:
        X_test, y_test = X_val, y_val

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description='Advanced Training with Robustness, Uncertainty, and Multiclass')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory containing training data')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='Image size (height width)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='attention_unet',
                        choices=['unet', 'attention_unet', 'res_unet'],
                        help='Type of model to train')
    parser.add_argument('--base_filters', type=int, default=64,
                        help='Number of base filters in model')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    
    # Task-specific arguments
    parser.add_argument('--task', type=str, default='robustness',
                        choices=['robustness', 'uncertainty', 'multiclass'],
                        help='Task to perform')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes for multiclass segmentation')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of MC samples for uncertainty estimation')
    parser.add_argument('--corruption_types', type=str, nargs='+',
                        default=['gaussian_noise', 'salt_pepper_noise', 'gaussian_blur',
                                'brightness', 'contrast', 'rotation'],
                        help='Types of corruptions for robustness analysis')
    
    # General arguments
    parser.add_argument('--save_dir', type=str, default='./advanced_results',
                        help='Directory to save models and results')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    config = vars(args)
    
    # Set random seeds
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Run appropriate task
    if args.task == 'robustness':
        model, results = train_with_robustness_analysis(config)
    elif args.task == 'uncertainty':
        model, mean_pred, uncertainty = train_with_uncertainty_estimation(config)
    elif args.task == 'multiclass':
        model, metrics = train_multiclass_model(config)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Results saved to {args.save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()