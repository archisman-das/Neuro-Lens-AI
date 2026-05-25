"""
Advanced Training Script with Robustness Analysis, Uncertainty Estimation, and Multiclass Classification
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import os
from datetime import datetime

from segmentation_models import build_unet, build_attention_unet, dice_loss, combined_loss
from robustness_analysis import RobustnessAnalyzer, UncertaintyEstimator, MulticlassSegmentationModel
from kfold_validation import SegmentationKFoldValidator


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


def load_data(config):
    """
    Load binary segmentation data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Placeholder - implement based on your data structure
    # For now, return dummy data
    image_size = tuple(config.get('image_size', [224, 224]))
    
    # Generate dummy data for testing
    X = np.random.rand(100, *image_size, 3).astype(np.float32)
    y = (np.random.rand(100, *image_size, 1) > 0.7).astype(np.float32)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_multiclass_data(config):
    """
    Load multiclass segmentation data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Placeholder - implement based on your data structure
    image_size = tuple(config.get('image_size', [224, 224]))
    num_classes = config.get('num_classes', 4)
    
    # Generate dummy data for testing
    X = np.random.rand(200, *image_size, 3).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(200, *image_size)).astype(np.int32)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
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