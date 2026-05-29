"""
Training Script for Brain Tumor Segmentation Models
"""

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from src.segmentation_models import (
    build_unet,
    build_attention_unet,
    build_res_unet,
    build_multi_modal_unet,
    dice_coefficient,
    dice_loss,
    combined_loss,
    iou_metric,
)
from src.kfold_validation import SegmentationKFoldValidator, prepare_data_for_kfold
from src.ablation_study import (
    SegmentationAblationStudy,
    calculate_segmentation_metrics,
    create_attention_ablation_study,
    create_architecture_ablation_study,
    create_loss_ablation_study,
)


def get_model(config):
    """
    Build model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compiled model
    """
    model_type = config.get('model_type', 'unet')
    input_shape = config.get('input_shape', (224, 224, 3))
    num_classes = config.get('num_classes', 1)
    base_filters = config.get('base_filters', 64)
    dropout_rate = config.get('dropout_rate', 0.2)
    use_attention = config.get('use_attention', False)
    
    if model_type == 'unet':
        model = build_unet(
            input_shape=input_shape,
            num_classes=num_classes,
            base_filters=base_filters,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
    elif model_type == 'attention_unet':
        model = build_attention_unet(
            input_shape=input_shape,
            num_classes=num_classes,
            base_filters=base_filters,
            dropout_rate=dropout_rate
        )
    elif model_type == 'res_unet':
        model = build_res_unet(
            input_shape=input_shape,
            num_classes=num_classes,
            base_filters=base_filters,
            dropout_rate=dropout_rate
        )
    elif model_type == 'multi_modal_unet':
        input_shapes = config.get('input_shapes', [(224, 224, 3), (224, 224, 3)])
        fusion_method = config.get('fusion_method', 'attention')
        model = build_multi_modal_unet(
            input_shapes=input_shapes,
            num_classes=num_classes,
            base_filters=base_filters,
            dropout_rate=dropout_rate,
            fusion_method=fusion_method
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def compile_model(model, config):
    """
    Compile model with loss function and metrics
    
    Args:
        model: Model to compile
        config: Configuration dictionary
    """
    loss_fn = config.get('loss_fn', 'dice_bce')
    learning_rate = config.get('learning_rate', 1e-4)
    
    # Get loss function
    if loss_fn == 'dice_bce':
        loss = combined_loss(weights=[0.5, 0.5])
    elif loss_fn == 'dice':
        loss = dice_loss
    elif loss_fn == 'bce':
        loss = 'binary_crossentropy'
    elif loss_fn == 'focal':
        gamma = config.get('focal_gamma', 2.0)
        def focal_loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            focal_weight = tf.pow(1 - y_pred, gamma) * y_true + tf.pow(y_pred, gamma) * (1 - y_true)
            return cross_entropy * focal_weight
        loss = focal_loss
    else:
        loss = 'binary_crossentropy'
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            dice_coefficient,
            iou_metric,
            'accuracy'
        ]
    )
    
    return model


def load_data(config):
    """
    Load training data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (images, masks) or (X_train, y_train, X_val, y_val)
    """
    data_dir = Path(config.get('data_dir', './dataset'))
    image_size = tuple(config.get('image_size', (224, 224)))
    
    # Check for pre-split data
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    if train_dir.exists() and val_dir.exists():
        # Load pre-split data
        X_train, y_train = load_images_and_masks(train_dir, image_size)
        X_val, y_val = load_images_and_masks(val_dir, image_size)
        return X_train, y_train, X_val, y_val
    else:
        # Load all data and split
        images, masks = load_images_and_masks(data_dir, image_size)
        return images, masks


def load_images_and_masks(data_dir, image_size):
    """
    Load images and masks from directory
    
    Args:
        data_dir: Directory containing images and masks subdirectories
        image_size: Size to resize images to
        
    Returns:
        Tuple of (images array, masks array)
    """
    import cv2
    
    images_dir = Path(data_dir) / 'images'
    masks_dir = Path(data_dir) / 'masks'
    
    if not images_dir.exists():
        # Try loading directly from data_dir
        images_dir = Path(data_dir)
        masks_dir = Path(data_dir)
    
    # Get file lists
    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    mask_files = sorted(list(masks_dir.glob('*.jpg')) + list(masks_dir.glob('*.png')))
    
    if len(image_files) != len(mask_files):
        raise ValueError(f"Mismatch between images ({len(image_files)}) and masks ({len(mask_files)})")
    
    # Load images
    images = []
    masks = []
    
    for img_path, mask_path in zip(image_files, mask_files):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        images.append(img)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
    
    return np.array(images), np.array(masks)


def create_callbacks(config, save_dir):
    """
    Create training callbacks
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save model checkpoints
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('patience', 15),
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Model checkpoint
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    )
    
    # Learning rate scheduler
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    
    callbacks.append(
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    )
    
    # TensorBoard
    if config.get('use_tensorboard', False):
        log_dir = os.path.join(save_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        )
    
    # CSV logger
    callbacks.append(
        tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'training_history.csv'))
    )
    
    return callbacks


def train_model(config):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Trained model and history
    """
    # Set random seeds
    np.random.seed(config.get('random_seed', 42))
    tf.random.set_seed(config.get('random_seed', 42))
    
    # Create save directory
    save_dir = Path(config.get('save_dir', './segmentation_models'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    print("Loading data...")
    data = load_data(config)
    
    if len(data) == 2:
        # Need to split data
        images, masks = data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            images, masks, test_size=config.get('val_split', 0.2), random_state=config.get('random_seed', 42)
        )
    else:
        X_train, y_train, X_val, y_val = data
    
    print(f"Training data: {X_train.shape[0]} images")
    print(f"Validation data: {X_val.shape[0]} images")
    
    # Build and compile model
    print("Building model...")
    model = get_model(config)
    model = compile_model(model, config)
    
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(config, str(save_dir))
    
    # Train
    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 16),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(save_dir / 'final_model.h5')
    
    # Plot training history
    plot_training_history(history, save_dir / 'training_history.png')
    
    # Evaluate
    print("Evaluating model...")
    eval_results = model.evaluate(X_val, y_val, verbose=0)
    
    # Save evaluation results
    eval_dict = {
        metric_name: float(value)
        for metric_name, value in zip(model.metrics_names, eval_results)
    }
    
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_dict, f, indent=2)
    
    print(f"Evaluation results: {eval_dict}")
    
    return model, history


def train_with_kfold(config):
    """
    Train model with k-fold cross-validation
    
    Args:
        config: Configuration dictionary
        
    Returns:
        KFoldValidator with trained models
    """
    # Set random seeds
    np.random.seed(config.get('random_seed', 42))
    tf.random.set_seed(config.get('random_seed', 42))
    
    # Create save directory
    save_dir = Path(config.get('save_dir', './kfold_segmentation_results'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    images, masks = load_data(config)
    
    # Create model builder function
    def model_builder():
        model = get_model(config)
        return compile_model(model, config)
    
    # Create K-fold validator
    validator = SegmentationKFoldValidator(
        model_builder=model_builder,
        n_splits=config.get('n_splits', 5),
        shuffle=config.get('shuffle', True),
        random_state=config.get('random_seed', 42),
        image_size=tuple(config.get('image_size', (224, 224)))
    )
    
    # Run cross-validation
    results = validator.cross_validate(
        images=images,
        masks=masks,
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 16),
        save_dir=str(save_dir),
        augment=config.get('use_augmentation', True)
    )
    
    return validator, results


def run_ablation_study(config):
    """
    Run ablation study on segmentation models
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AblationStudy with results
    """
    # Load data
    print("Loading data...")
    images, masks = load_data(config)
    
    # Split into train/val for ablation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=config.get('random_seed', 42)
    )
    
    data = (X_train, y_train, X_val, y_val)
    
    # Create ablation study
    base_config = {
        'input_shape': tuple(config.get('image_size', (224, 224))) + (3,),
        'num_classes': 1,
        'base_filters': config.get('base_filters', 64),
        'dropout_rate': config.get('dropout_rate', 0.2),
        'learning_rate': config.get('learning_rate', 1e-4),
        'epochs': config.get('epochs', 50),
        'batch_size': config.get('batch_size', 16)
    }
    
    # Choose ablation type
    ablation_type = config.get('ablation_type', 'attention')
    
    if ablation_type == 'attention':
        study = create_attention_ablation_study(
            base_config,
            results_dir=config.get('ablation_results_dir', './attention_ablation')
        )
    elif ablation_type == 'architecture':
        study = create_architecture_ablation_study(
            base_config,
            results_dir=config.get('ablation_results_dir', './architecture_ablation')
        )
    elif ablation_type == 'loss':
        study = create_loss_ablation_study(
            base_config,
            results_dir=config.get('ablation_results_dir', './loss_ablation')
        )
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    # Define model builder
    def model_builder(cfg):
        model = build_unet(
            input_shape=cfg.get('input_shape', (224, 224, 3)),
            num_classes=cfg.get('num_classes', 1),
            base_filters=cfg.get('base_filters', 64),
            dropout_rate=cfg.get('dropout_rate', 0.2),
            use_attention=cfg.get('use_attention', False)
        )
        return compile_model(model, cfg)
    
    # Run ablation study
    results = study.run_all_experiments(
        model_builder=model_builder,
        data=data,
        metrics_calculator=calculate_segmentation_metrics
    )
    
    return study, results


def plot_training_history(history, save_path):
    """
    Plot training history
    
    Args:
        history: Training history object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Dice coefficient
    if 'dice_coefficient' in history.history:
        axes[1].plot(history.history['dice_coefficient'], label='Train Dice')
        if 'val_dice_coefficient' in history.history:
            axes[1].plot(history.history['val_dice_coefficient'], label='Val Dice')
        axes[1].set_title('Dice Coefficient')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice')
        axes[1].legend()
    
    # IoU
    if 'iou_metric' in history.history:
        axes[2].plot(history.history['iou_metric'], label='Train IoU')
        if 'val_iou_metric' in history.history:
            axes[2].plot(history.history['val_iou_metric'], label='Val IoU')
        axes[2].set_title('Intersection over Union')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('IoU')
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Brain Tumor Segmentation Models')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory containing training data')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='Image size (height width)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'attention_unet', 'res_unet', 'multi_modal_unet'],
                        help='Type of model to train')
    parser.add_argument('--base_filters', type=int, default=64,
                        help='Number of base filters in model')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--use_attention', action='store_true',
                        help='Use attention gates in U-Net')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--loss_fn', type=str, default='dice_bce',
                        choices=['dice_bce', 'dice', 'bce', 'focal'],
                        help='Loss function')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # K-fold arguments
    parser.add_argument('--use_kfold', action='store_true',
                        help='Use k-fold cross-validation')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    # Ablation study arguments
    parser.add_argument('--use_ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--ablation_type', type=str, default='attention',
                        choices=['attention', 'architecture', 'loss'],
                        help='Type of ablation study')
    
    # General arguments
    parser.add_argument('--save_dir', type=str, default='./segmentation_models',
                        help='Directory to save models and results')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use TensorBoard logging')
    
    args = parser.parse_args()
    config = vars(args)
    
    # Run appropriate training mode
    if args.use_kfold:
        validator, results = train_with_kfold(config)
        print("\nK-Fold Cross-Validation Results:")
        print(f"Mean validation loss: {results['aggregate_metrics']['val_loss']['mean']:.4f} ± {results['aggregate_metrics']['val_loss']['std']:.4f}")
    elif args.use_ablation:
        study, results = run_ablation_study(config)
        print("\nAblation Study Results:")
        print(study.get_comparison_table())
    else:
        model, history = train_model(config)
        print("\nTraining completed!")


if __name__ == '__main__':
    main()