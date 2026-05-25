"""
Ablation Study Framework for Brain Tumor Detection Models
"""

import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class AblationStudy:
    """
    Framework for conducting ablation studies on brain tumor detection models
    """
    
    def __init__(self, base_config, results_dir='./ablation_results'):
        """
        Initialize ablation study
        
        Args:
            base_config: Base configuration dictionary
            results_dir: Directory to save results
        """
        self.base_config = base_config
        self.results_dir = results_dir
        self.results = {}
        self.study_metadata = {
            'start_time': datetime.now().isoformat(),
            'base_config': base_config,
            'experiments': []
        }
        
        os.makedirs(results_dir, exist_ok=True)
    
    def add_experiment(self, name, config_modification, description=""):
        """
        Add an experiment configuration
        
        Args:
            name: Name of the experiment
            config_modification: Dictionary of config modifications
            description: Description of what this experiment tests
        """
        config = self.base_config.copy()
        config.update(config_modification)
        
        self.study_metadata['experiments'].append({
            'name': name,
            'config': config,
            'description': description,
            'modifications': config_modification
        })
    
    def run_experiment(self, experiment_idx, model_builder, data, metrics_calculator):
        """
        Run a single ablation experiment
        
        Args:
            experiment_idx: Index of experiment in study_metadata['experiments']
            model_builder: Function to build model with given config
            data: Tuple of (X_train, y_train, X_val, y_val)
            metrics_calculator: Function to calculate metrics
            
        Returns:
            Results dictionary
        """
        experiment = self.study_metadata['experiments'][experiment_idx]
        config = experiment['config']
        name = experiment['name']
        
        print(f"\n{'='*60}")
        print(f"Running Ablation Experiment: {name}")
        print(f"{'='*60}")
        print(f"Description: {experiment['description']}")
        print(f"Modifications: {experiment['modifications']}")
        
        # Build model
        model = model_builder(config)
        
        # Train model
        history = self._train_model(model, data, config)
        
        # Evaluate
        metrics = metrics_calculator(model, data)
        
        # Store results
        self.results[name] = {
            'metrics': metrics,
            'history': history.history if hasattr(history, 'history') else history,
            'config': config
        }
        
        print(f"Experiment {name} completed. Metrics: {metrics}")
        
        return self.results[name]
    
    def _train_model(self, model, data, config):
        """
        Train model with given configuration
        
        Args:
            model: Model to train
            data: Training data tuple
            config: Training configuration
            
        Returns:
            Training history
        """
        X_train, y_train, X_val, y_val = data
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 1e-4)),
            loss=config.get('loss_fn', 'binary_crossentropy'),
            metrics=config.get('metrics', ['accuracy'])
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.get('patience', 10),
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def run_all_experiments(self, model_builder, data, metrics_calculator):
        """
        Run all experiments in the study
        
        Args:
            model_builder: Function to build model with given config
            data: Training data tuple
            metrics_calculator: Function to calculate metrics
        """
        for i in range(len(self.study_metadata['experiments'])):
            self.run_experiment(i, model_builder, data, metrics_calculator)
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save ablation study results"""
        # Save summary
        summary_path = os.path.join(self.results_dir, 'ablation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'study_metadata': self.study_metadata,
                'results': self.results
            }, f, indent=2)
        
        # Save detailed results as CSV
        results_data = []
        for name, result in self.results.items():
            row = {'experiment': name}
            row.update(result['metrics'])
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(os.path.join(self.results_dir, 'ablation_results.csv'), index=False)
        
        # Save plots
        self.plot_results()
        
        print(f"Results saved to {self.results_dir}")
    
    def plot_results(self):
        """Plot ablation study results"""
        if not self.results:
            return
        
        # Extract metrics
        experiments = list(self.results.keys())
        metrics_names = list(list(self.results.values())[0]['metrics'].keys())
        
        # Create subplots for each metric
        fig, axes = plt.subplots(1, len(metrics_names), figsize=(6*len(metrics_names), 5))
        if len(metrics_names) == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, metrics_names):
            values = [self.results[exp]['metrics'][metric_name] for exp in experiments]
            
            # Create bar plot
            bars = ax.bar(experiments, values, color=plt.cm.Set3(np.linspace(0, 1, len(experiments))))
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'ablation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot training histories
        if len(experiments) > 0 and 'history' in self.results[experiments[0]]:
            self._plot_training_histories(experiments)
    
    def _plot_training_histories(self, experiments):
        """Plot training histories for all experiments"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for exp in experiments:
            history = self.results[exp]['history']
            if 'loss' in history:
                axes[0].plot(history['loss'], label=f'{exp} - train')
                if 'val_loss' in history:
                    axes[0].plot(history['val_loss'], label=f'{exp} - val', linestyle='--')
            
            if 'accuracy' in history:
                axes[1].plot(history['accuracy'], label=f'{exp} - train')
                if 'val_accuracy' in history:
                    axes[1].plot(history['val_accuracy'], label=f'{exp} - val', linestyle='--')
        
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_histories.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_comparison_table(self):
        """
        Get comparison table of all experiments
        
        Returns:
            Pandas DataFrame with comparison results
        """
        if not self.results:
            raise ValueError("No results available. Run experiments first.")
        
        rows = []
        for name, result in self.results.items():
            row = {'Experiment': name}
            row.update(result['metrics'])
            row['Description'] = next(
                (exp['description'] for exp in self.study_metadata['experiments'] 
                 if exp['name'] == name), ''
            )
            rows.append(row)
        
        return pd.DataFrame(rows)


class SegmentationAblationStudy(AblationStudy):
    """
    Ablation study framework specifically for segmentation models
    """
    
    def __init__(self, base_config, results_dir='./segmentation_ablation_results'):
        super().__init__(base_config, results_dir)
    
    def add_segmentation_experiment(self, name, model_config, training_config, description=""):
        """
        Add a segmentation experiment
        
        Args:
            name: Experiment name
            model_config: Model configuration modifications
            training_config: Training configuration modifications
            description: Description of the experiment
        """
        config = {
            **self.base_config,
            **model_config,
            **training_config
        }
        
        self.study_metadata['experiments'].append({
            'name': name,
            'config': config,
            'description': description,
            'model_modifications': model_config,
            'training_modifications': training_config
        })
    
    def run_segmentation_experiment(self, experiment_idx, model_builder, data, metrics_calculator):
        """
        Run a segmentation experiment
        
        Args:
            experiment_idx: Index of experiment
            model_builder: Function to build model
            data: Tuple of (X_train, y_train, X_val, y_val) where y are masks
            metrics_calculator: Function to calculate segmentation metrics
            
        Returns:
            Results dictionary
        """
        experiment = self.study_metadata['experiments'][experiment_idx]
        config = experiment['config']
        name = experiment['name']
        
        print(f"\n{'='*60}")
        print(f"Running Segmentation Ablation: {name}")
        print(f"{'='*60}")
        
        # Build model
        model = model_builder(config)
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 1e-4)),
            loss=config.get('loss_fn', 'binary_crossentropy'),
            metrics=[
                tf.keras.metrics.MeanIoU(num_classes=2),
                'accuracy'
            ]
        )
        
        # Train
        X_train, y_train, X_val, y_val = data
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.get('patience', 10),
                restore_best_weights=True
            )
        ]
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 16),
            callbacks=callbacks
        )
        
        # Evaluate with custom metrics
        metrics = metrics_calculator(model, (X_val, y_val))
        
        # Store results
        self.results[name] = {
            'metrics': metrics,
            'history': history.history,
            'config': config
        }
        
        print(f"Segmentation ablation {name} completed. Metrics: {metrics}")
        
        return self.results[name]


def create_attention_ablation_study(base_config, results_dir='./attention_ablation'):
    """
    Create an ablation study for attention mechanisms
    
    Args:
        base_config: Base configuration
        results_dir: Directory to save results
        
    Returns:
        AblationStudy instance with experiments added
    """
    study = AblationStudy(base_config, results_dir)
    
    # Baseline without attention
    study.add_experiment(
        name='baseline_no_attention',
        config_modification={'use_attention': False},
        description='Baseline model without any attention mechanisms'
    )
    
    # With attention gates in skip connections
    study.add_experiment(
        name='attention_skip_connections',
        config_modification={'use_attention': True},
        description='Model with attention gates in U-Net skip connections'
    )
    
    # With channel attention
    study.add_experiment(
        name='channel_attention',
        config_modification={'attention_type': 'channel'},
        description='Model with channel-wise attention mechanism'
    )
    
    # With spatial attention
    study.add_experiment(
        name='spatial_attention',
        config_modification={'attention_type': 'spatial'},
        description='Model with spatial attention mechanism'
    )
    
    # With both channel and spatial attention
    study.add_experiment(
        name='cbam_attention',
        config_modification={'attention_type': 'cbam'},
        description='Model with combined channel and spatial attention (CBAM)'
    )
    
    return study


def create_architecture_ablation_study(base_config, results_dir='./architecture_ablation'):
    """
    Create an ablation study for architecture variations
    
    Args:
        base_config: Base configuration
        results_dir: Directory to save results
        
    Returns:
        AblationStudy instance with experiments added
    """
    study = AblationStudy(base_config, results_dir)
    
    # Baseline
    study.add_experiment(
        name='baseline',
        config_modification={},
        description='Baseline architecture'
    )
    
    # Different depths
    study.add_experiment(
        name='shallow_network',
        config_modification={'num_layers': 3},
        description='Shallower network with fewer layers'
    )
    
    study.add_experiment(
        name='deep_network',
        config_modification={'num_layers': 6},
        description='Deeper network with more layers'
    )
    
    # Different filter sizes
    study.add_experiment(
        name='smaller_filters',
        config_modification={'base_filters': 32},
        description='Network with smaller base number of filters'
    )
    
    study.add_experiment(
        name='larger_filters',
        config_modification={'base_filters': 128},
        description='Network with larger base number of filters'
    )
    
    # With residual connections
    study.add_experiment(
        name='residual_connections',
        config_modification={'use_residual': True},
        description='Network with residual connections'
    )
    
    # With dense connections
    study.add_experiment(
        name='dense_connections',
        config_modification={'use_dense': True},
        description='Network with dense connections'
    )
    
    return study


def create_loss_ablation_study(base_config, results_dir='./loss_ablation'):
    """
    Create an ablation study for different loss functions
    
    Args:
        base_config: Base configuration
        results_dir: Directory to save results
        
    Returns:
        AblationStudy instance with experiments added
    """
    study = AblationStudy(base_config, results_dir)
    
    # Baseline cross-entropy
    study.add_experiment(
        name='cross_entropy',
        config_modification={'loss_fn': 'binary_crossentropy'},
        description='Standard binary cross-entropy loss'
    )
    
    # Dice loss
    study.add_experiment(
        name='dice_loss',
        config_modification={'loss_fn': 'dice_loss'},
        description='Dice loss for better handling of class imbalance'
    )
    
    # Combined loss
    study.add_experiment(
        name='combined_dice_bce',
        config_modification={'loss_fn': 'combined_dice_bce', 'loss_weights': [0.5, 0.5]},
        description='Combined Dice and BCE loss'
    )
    
    # Focal loss
    study.add_experiment(
        name='focal_loss',
        config_modification={'loss_fn': 'focal_loss', 'focal_gamma': 2.0},
        description='Focal loss for hard example mining'
    )
    
    # Tversky loss
    study.add_experiment(
        name='tversky_loss',
        config_modification={'loss_fn': 'tversky_loss', 'alpha': 0.5, 'beta': 0.5},
        description='Tversky loss for imbalanced segmentation'
    )
    
    return study


def create_data_augmentation_ablation_study(base_config, results_dir='./augmentation_ablation'):
    """
    Create an ablation study for data augmentation strategies
    
    Args:
        base_config: Base configuration
        results_dir: Directory to save results
        
    Returns:
        AblationStudy instance with experiments added
    """
    study = AblationStudy(base_config, results_dir)
    
    # No augmentation
    study.add_experiment(
        name='no_augmentation',
        config_modification={'use_augmentation': False},
        description='No data augmentation'
    )
    
    # Basic augmentation
    study.add_experiment(
        name='basic_augmentation',
        config_modification={
            'use_augmentation': True,
            'augmentation': ['flip', 'rotation']
        },
        description='Basic augmentation: flips and rotations'
    )
    
    # Advanced augmentation
    study.add_experiment(
        name='advanced_augmentation',
        config_modification={
            'use_augmentation': True,
            'augmentation': ['flip', 'rotation', 'zoom', 'contrast', 'brightness']
        },
        description='Advanced augmentation with multiple transformations'
    )
    
    # With MixUp
    study.add_experiment(
        name='mixup_augmentation',
        config_modification={
            'use_augmentation': True,
            'augmentation': ['flip', 'rotation', 'mixup'],
            'mixup_alpha': 0.2
        },
        description='Augmentation with MixUp strategy'
    )
    
    # With CutMix
    study.add_experiment(
        name='cutmix_augmentation',
        config_modification={
            'use_augmentation': True,
            'augmentation': ['flip', 'rotation', 'cutmix'],
            'cutmix_alpha': 1.0
        },
        description='Augmentation with CutMix strategy'
    )
    
    return study


def calculate_segmentation_metrics(model, data, thresholds=None):
    """
    Calculate comprehensive segmentation metrics
    
    Args:
        model: Trained segmentation model
        data: Tuple of (X_val, y_val)
        thresholds: List of thresholds for binary classification
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import dice_score, jaccard_score, confusion_matrix
    
    X_val, y_val = data
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Use threshold of 0.5 by default
    if thresholds is None:
        thresholds = [0.5]
    
    metrics = {}
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Flatten for metric calculation
        y_val_flat = y_val.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        # Dice coefficient
        dice = dice_score(y_val_flat, y_pred_flat)
        
        # IoU (Jaccard index)
        iou = jaccard_score(y_val_flat, y_pred_flat)
        
        # Precision, Recall, F1
        tn, fp, fn, tp = confusion_matrix(y_val_flat, y_pred_flat).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics[f'dice_t{threshold}'] = float(dice)
        metrics[f'iou_t{threshold}'] = float(iou)
        metrics[f'precision_t{threshold}'] = float(precision)
        metrics[f'recall_t{threshold}'] = float(recall)
        metrics[f'f1_t{threshold}'] = float(f1)
        metrics[f'specificity_t{threshold}'] = float(specificity)
    
    return metrics