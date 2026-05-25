"""
Robustness Analysis and Uncertainty Estimation for Brain Tumor Segmentation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


class RobustnessAnalyzer:
    """
    Analyze model robustness to various perturbations and corruptions
    """
    
    def __init__(self, model, input_shape=(224, 224, 3)):
        """
        Initialize robustness analyzer
        
        Args:
            model: Trained segmentation model
            input_shape: Shape of input images
        """
        self.model = model
        self.input_shape = input_shape
        self.results = {}
    
    def add_gaussian_noise(self, image, std=0.1):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 1)
    
    def add_salt_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        """Add salt and pepper noise to image"""
        noisy_image = image.copy()
        
        # Salt noise
        salt_mask = np.random.random(image.shape) < salt_prob
        noisy_image[salt_mask] = 1
        
        # Pepper noise
        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def add_gaussian_blur(self, image, kernel_size=3):
        """Apply Gaussian blur to image"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=kernel_size/3)
    
    def add_motion_blur(self, image, kernel_size=5, angle=0):
        """Apply motion blur to image"""
        from scipy.ndimage import convolve
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        kernel[center, :] = 1 / kernel_size
        
        # Rotate kernel
        from scipy.ndimage import rotate
        kernel = rotate(kernel, angle, reshape=False)
        
        # Apply convolution
        blurred = convolve(image, kernel, mode='reflect')
        return blurred
    
    def change_brightness(self, image, factor=0.5):
        """Change image brightness"""
        return np.clip(image * factor, 0, 1)
    
    def change_contrast(self, image, factor=0.5):
        """Change image contrast"""
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 1)
    
    def rotate_image(self, image, angle=10):
        """Rotate image by angle degrees"""
        from scipy.ndimage import rotate
        return rotate(image, angle, axes=(0, 1), reshape=False, mode='reflect')
    
    def scale_image(self, image, scale_factor=0.9):
        """Scale image by factor"""
        from scipy.ndimage import zoom
        h, w = image.shape[:2]
        scaled = zoom(image, (scale_factor, scale_factor, 1), order=1)
        
        # Crop or pad to original size
        if scaled.shape[0] > h:
            start = (scaled.shape[0] - h) // 2
            scaled = scaled[start:start+h, :, :]
        elif scaled.shape[0] < h:
            pad_h = (h - scaled.shape[0]) // 2
            scaled = np.pad(scaled, ((pad_h, h - scaled.shape[0] - pad_h), (0, 0), (0, 0)), mode='constant')
        
        if scaled.shape[1] > w:
            start = (scaled.shape[1] - w) // 2
            scaled = scaled[:, start:start+w, :]
        elif scaled.shape[1] < w:
            pad_w = (w - scaled.shape[1]) // 2
            scaled = np.pad(scaled, ((0, 0), (pad_w, w - scaled.shape[1] - pad_w), (0, 0)), mode='constant')
        
        return scaled
    
    def evaluate_robustness(self, X_test, y_test, corruption_type='gaussian_noise', 
                          corruption_levels=None, metric_fn=None):
        """
        Evaluate model robustness to a specific corruption type
        
        Args:
            X_test: Test images
            y_test: Ground truth masks
            corruption_type: Type of corruption to apply
            corruption_levels: List of corruption levels to test
            metric_fn: Function to compute metric (default: Dice coefficient)
            
        Returns:
            Dictionary of robustness metrics
        """
        if corruption_levels is None:
            corruption_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        
        if metric_fn is None:
            def metric_fn(y_true, y_pred):
                intersection = np.sum(y_true * y_pred)
                union = np.sum(y_true) + np.sum(y_pred)
                return (2. * intersection + 1e-6) / (union + 1e-6)
        
        # Get baseline performance
        baseline_preds = self.model.predict(X_test)
        baseline_score = np.mean([metric_fn(y_true, y_pred) 
                                 for y_true, y_pred in zip(y_test, baseline_preds)])
        
        # Get corruption function
        corruption_fn = getattr(self, f'add_{corruption_type}')
        
        # Evaluate at each corruption level
        scores = []
        for level in corruption_levels:
            # Apply corruption
            corrupted_X = np.array([corruption_fn(x, level) for x in X_test])
            
            # Predict on corrupted images
            preds = self.model.predict(corrupted_X)
            
            # Compute metric
            level_scores = [metric_fn(y_true, y_pred) 
                          for y_true, y_pred in zip(y_test, preds)]
            scores.append(np.mean(level_scores))
        
        # Compute robustness metrics
        results = {
            'corruption_type': corruption_type,
            'baseline_score': float(baseline_score),
            'corruption_levels': corruption_levels,
            'scores': [float(s) for s in scores],
            'mean_corruption_score': float(np.mean(scores)),
            'robustness_index': float(np.mean(scores) / baseline_score) if baseline_score > 0 else 0,
            'performance_drop': float(baseline_score - np.mean(scores))
        }
        
        self.results[corruption_type] = results
        
        return results
    
    def evaluate_all_corruptions(self, X_test, y_test, corruption_types=None, **kwargs):
        """
        Evaluate model robustness to all corruption types
        
        Args:
            X_test: Test images
            y_test: Ground truth masks
            corruption_types: List of corruption types to test
            **kwargs: Additional arguments for evaluate_robustness
            
        Returns:
            Dictionary of all robustness results
        """
        if corruption_types is None:
            corruption_types = [
                'gaussian_noise',
                'salt_pepper_noise',
                'gaussian_blur',
                'motion_blur',
                'brightness',
                'contrast',
                'rotation',
                'scaling'
            ]
        
        all_results = {}
        for corruption_type in corruption_types:
            print(f"Evaluating robustness to {corruption_type}...")
            results = self.evaluate_robustness(X_test, y_test, corruption_type, **kwargs)
            all_results[corruption_type] = results
        
        return all_results
    
    def plot_robustness_results(self, results=None, save_path=None):
        """
        Plot robustness analysis results
        
        Args:
            results: Results dictionary (if None, uses self.results)
            save_path: Path to save plot
        """
        if results is None:
            results = self.results
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, (corruption_type, result) in enumerate(results.items()):
            if idx >= 8:
                break
                
            ax = axes[idx]
            ax.plot(result['corruption_levels'], result['scores'], 'o-', linewidth=2)
            ax.axhline(y=result['baseline_score'], color='r', linestyle='--', alpha=0.5, label='Baseline')
            ax.set_title(f"{corruption_type.replace('_', ' ').title()}\n(Robustness Index: {result['robustness_index']:.3f})")
            ax.set_xlabel('Corruption Level')
            ax.set_ylabel('Dice Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results, save_dir='./robustness_results'):
        """Save robustness results to files"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as JSON
        with open(os.path.join(save_dir, 'robustness_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        rows = []
        for corruption_type, result in results.items():
            for level, score in zip(result['corruption_levels'], result['scores']):
                rows.append({
                    'corruption_type': corruption_type,
                    'corruption_level': level,
                    'dice_score': score,
                    'baseline_score': result['baseline_score'],
                    'robustness_index': result['robustness_index']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(save_dir, 'robustness_results.csv'), index=False)
        
        # Save plots
        self.plot_robustness_results(results, os.path.join(save_dir, 'robustness_plots.png'))


class UncertaintyEstimator:
    """
    Uncertainty estimation for segmentation models using Monte Carlo Dropout
    and Deep Ensembles
    """
    
    def __init__(self, model, num_samples=50, batch_size=32):
        """
        Initialize uncertainty estimator
        
        Args:
            model: Trained segmentation model with dropout
            num_samples: Number of Monte Carlo samples
            batch_size: Batch size for predictions
        """
        self.model = model
        self.num_samples = num_samples
        self.batch_size = batch_size
    
    def enable_dropout(self):
        """Enable dropout at inference time for MC Dropout"""
        for layer in self.model.layers:
            if isinstance(layer, layers.Dropout):
                layer.trainable = True
    
    def mc_dropout_predict(self, X, num_samples=None):
        """
        Monte Carlo Dropout prediction
        
        Args:
            X: Input images
            num_samples: Number of MC samples (if None, uses self.num_samples)
            
        Returns:
            mean_prediction, uncertainty, predictions_array
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        # Enable dropout
        self.enable_dropout()
        
        # Get multiple predictions with dropout enabled
        predictions = []
        for _ in range(num_samples):
            pred = self.model.predict(X, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty, predictions
    
    def deep_ensemble_predict(self, models, X):
        """
        Deep ensemble prediction
        
        Args:
            models: List of trained models
            X: Input images
            
        Returns:
            mean_prediction, uncertainty, predictions_array
        """
        predictions = []
        for model in models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty, predictions
    
    def compute_aleatoric_uncertainty(self, model, X, num_samples=10):
        """
        Estimate aleatoric uncertainty (data uncertainty)
        
        Args:
            model: Model that outputs both prediction and uncertainty
            X: Input images
            num_samples: Number of samples for estimation
            
        Returns:
            Aleatoric uncertainty map
        """
        # For models that output uncertainty directly
        predictions = []
        for _ in range(num_samples):
            pred = model.predict(X, verbose=0)
            if isinstance(pred, list) and len(pred) > 1:
                # Assume second output is uncertainty
                predictions.append(pred[1])
            else:
                predictions.append(pred)
        
        predictions = np.array(predictions)
        aleatoric_uncertainty = np.mean(predictions, axis=0)
        
        return aleatoric_uncertainty
    
    def compute_epistemic_uncertainty(self, models, X):
        """
        Estimate epistemic uncertainty (model uncertainty) using ensemble
        
        Args:
            models: List of trained models
            X: Input images
            
        Returns:
            Epistemic uncertainty map
        """
        predictions = []
        for model in models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        epistemic_uncertainty = np.var(predictions, axis=0)
        
        return epistemic_uncertainty
    
    def get_confidence_intervals(self, predictions_array, confidence_level=0.95):
        """
        Compute confidence intervals from prediction samples
        
        Args:
            predictions_array: Array of predictions (num_samples, height, width, channels)
            confidence_level: Confidence level for intervals
            
        Returns:
            lower_bound, upper_bound
        """
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions_array, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions_array, upper_percentile, axis=0)
        
        return lower_bound, upper_bound
    
    def visualize_uncertainty(self, image, mean_pred, uncertainty, threshold=0.5, 
                            save_path=None, cmap='viridis'):
        """
        Visualize prediction and uncertainty
        
        Args:
            image: Input image
            mean_pred: Mean prediction
            uncertainty: Uncertainty map
            threshold: Threshold for binary prediction
            save_path: Path to save visualization
            cmap: Colormap for uncertainty
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Input image
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Mean prediction
        binary_pred = (mean_pred >= threshold).astype(int)
        axes[1].imshow(binary_pred, cmap='gray')
        axes[1].set_title(f'Binary Prediction (threshold={threshold})')
        axes[1].axis('off')
        
        # Uncertainty map
        im = axes[2].imshow(uncertainty, cmap=cmap)
        axes[2].set_title('Uncertainty Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        # Overlay uncertainty on prediction
        axes[3].imshow(binary_pred, cmap='gray', alpha=0.7)
        im = axes[3].imshow(uncertainty, cmap=cmap, alpha=0.5)
        axes[3].set_title('Prediction with Uncertainty Overlay')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_uncertainty_results(self, mean_pred, uncertainty, image, save_dir='./uncertainty_results'):
        """Save uncertainty estimation results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save predictions
        np.save(os.path.join(save_dir, 'mean_prediction.npy'), mean_pred)
        np.save(os.path.join(save_dir, 'uncertainty.npy'), uncertainty)
        
        # Save visualization
        self.visualize_uncertainty(
            image, mean_pred, uncertainty,
            save_path=os.path.join(save_dir, 'uncertainty_visualization.png')
        )
        
        # Save summary statistics
        stats = {
            'mean_uncertainty': float(np.mean(uncertainty)),
            'max_uncertainty': float(np.max(uncertainty)),
            'min_uncertainty': float(np.min(uncertainty)),
            'std_uncertainty': float(np.std(uncertainty)),
            'high_uncertainty_pixels': float(np.mean(uncertainty > 0.5)),
            'medium_uncertainty_pixels': float(np.mean((uncertainty > 0.2) & (uncertainty <= 0.5))),
            'low_uncertainty_pixels': float(np.mean(uncertainty <= 0.2))
        }
        
        with open(os.path.join(save_dir, 'uncertainty_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)


class MulticlassSegmentationModel:
    """
    Multiclass segmentation model for different tumor types
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4, base_filters=64, dropout_rate=0.2):
        """
        Initialize multiclass segmentation model
        
        Args:
            input_shape: Shape of input images
            num_classes: Number of segmentation classes (including background)
            base_filters: Number of base filters
            dropout_rate: Dropout rate
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate
        self.model = None
        
        # Class names (can be customized)
        self.class_names = ['background', 'glioma', 'meningioma', 'pituitary']
    
    def build_model(self, use_attention=False):
        """
        Build multiclass U-Net model
        
        Args:
            use_attention: Whether to use attention gates
            
        Returns:
            Compiled model
        """
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # Normalize input
        x = layers.Rescaling(1.0 / 255)(inputs)
        
        # Encoder
        filters = self.base_filters
        skip_connections = []
        
        for i in range(4):
            # Convolutional block
            x = layers.Conv2D(filters, 3, activation='relu', padding='same', 
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.Conv2D(filters, 3, activation='relu', padding='same', 
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            
            skip_connections.append(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            filters *= 2
        
        # Bottleneck
        x = layers.Conv2D(filters, 3, activation='relu', padding='same', 
                        kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same', 
                        kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder
        filters //= 2
        
        for i in range(4):
            # Upsampling
            x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
            
            # Apply attention gate if enabled
            if use_attention:
                # Attention mechanism
                skip = skip_connections.pop()
                attention_weights = layers.Conv2D(filters, 1, padding='same', use_bias=False)(skip)
                attention_weights = layers.Activation('sigmoid')(attention_weights)
                x = layers.Concatenate()([x, layers.Multiply()([skip, attention_weights])])
            else:
                # Simple concatenation
                x = layers.Concatenate()([x, skip_connections.pop()])
            
            # Convolutional block
            x = layers.Conv2D(filters, 3, activation='relu', padding='same', 
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.Conv2D(filters, 3, activation='relu', padding='same', 
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            
            filters //= 2
        
        # Output layer with softmax for multiclass
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax', padding='same')(x)
        
        self.model = Model(inputs=[inputs], outputs=[outputs], name='multiclass_unet')
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compile model with appropriate loss and metrics
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
        
        # Categorical crossentropy for multiclass
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.MeanIoU(num_classes=self.num_classes),
                self.dice_coefficient_multiclass
            ]
        )
        
        return self.model
    
    def dice_coefficient_multiclass(self, y_true, y_pred, smooth=1e-6):
        """
        Dice coefficient for multiclass segmentation
        
        Args:
            y_true: Ground truth one-hot encoded masks
            y_pred: Predicted probabilities
            smooth: Smoothing factor
            
        Returns:
            Mean Dice coefficient across classes
        """
        # Get class predictions
        y_true_classes = tf.argmax(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # One-hot encode predictions
        y_true_onehot = tf.one_hot(y_true_classes, depth=self.num_classes)
        y_pred_onehot = tf.one_hot(y_pred_classes, depth=self.num_classes)
        
        # Compute Dice for each class
        dice_scores = []
        for i in range(self.num_classes):
            intersection = tf.reduce_sum(y_true_onehot[..., i] * y_pred_onehot[..., i])
            union = tf.reduce_sum(y_true_onehot[..., i]) + tf.reduce_sum(y_pred_onehot[..., i])
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        return tf.reduce_mean(dice_scores)
    
    def prepare_multiclass_masks(self, masks, num_classes=None):
        """
        Convert integer masks to one-hot encoded masks
        
        Args:
            masks: Integer masks with values 0 to num_classes-1
            num_classes: Number of classes (if None, uses self.num_classes)
            
        Returns:
            One-hot encoded masks
        """
        if num_classes is None:
            num_classes = self.num_classes
        
        # Convert to one-hot
        one_hot = tf.one_hot(masks.astype(int), depth=num_classes)
        
        return one_hot.numpy()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=16, callbacks=None):
        """
        Train the multiclass model
        
        Args:
            X_train: Training images
            y_train: Training masks (integer or one-hot)
            X_val: Validation images
            y_val: Validation masks
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        # Build and compile model
        self.compile_model()
        
        # Convert masks to one-hot if needed
        if len(y_train.shape) == 3 or (len(y_train.shape) == 4 and y_train.shape[-1] != self.num_classes):
            y_train = self.prepare_multiclass_masks(y_train)
            y_val = self.prepare_multiclass_masks(y_val)
        
        # Default callbacks
        if callbacks is None:
            callbacks = []
        
        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ])
        
        # Train
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Predict segmentation masks
        
        Args:
            X: Input images
            
        Returns:
            Predicted masks (integer format)
        """
        # Get predictions
        predictions = self.model.predict(X)
        
        # Convert to integer masks
        predicted_masks = np.argmax(predictions, axis=-1)
        
        return predicted_masks
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Input images
            
        Returns:
            Probability maps for each class
        """
        return self.model.predict(X)
    
    def evaluate_multiclass(self, X_test, y_test, class_names=None):
        """
        Evaluate multiclass segmentation with per-class metrics
        
        Args:
            X_test: Test images
            y_test: Ground truth masks (integer format)
            class_names: List of class names
            
        Returns:
            Dictionary of evaluation metrics
        """
        if class_names is None:
            class_names = self.class_names
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Flatten for metric calculation
        y_true_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Compute per-class metrics
        metrics = {}
        
        # Overall metrics
        overall_accuracy = np.mean(y_true_flat == y_pred_flat)
        
        # Per-class metrics
        class_metrics = {}
        for class_idx, class_name in enumerate(class_names):
            # Binary mask for this class
            true_binary = (y_true_flat == class_idx)
            pred_binary = (y_pred_flat == class_idx)
            
            # Compute metrics
            intersection = np.sum(true_binary & pred_binary)
            union = np.sum(true_binary) + np.sum(pred_binary) - intersection
            
            if np.sum(true_binary) > 0:
                recall = intersection / np.sum(true_binary)
            else:
                recall = 0
            
            if np.sum(pred_binary) > 0:
                precision = intersection / np.sum(pred_binary)
            else:
                precision = 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0
            
            dice = (2 * intersection + 1e-6) / (np.sum(true_binary) + np.sum(pred_binary) + 1e-6)
            
            class_metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'iou': float(iou),
                'dice': float(dice),
                'support': int(np.sum(true_binary))
            }
        
        # Compute mean metrics
        mean_metrics = {}
        for metric in ['precision', 'recall', 'f1_score', 'iou', 'dice']:
            mean_metrics[f'mean_{metric}'] = float(np.mean([cm[metric] for cm in class_metrics.values()]))
        
        metrics['overall_accuracy'] = float(overall_accuracy)
        metrics['class_metrics'] = class_metrics
        metrics['mean_metrics'] = mean_metrics
        
        # Confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(len(class_names))))
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def visualize_multiclass_prediction(self, image, true_mask, pred_mask, class_names=None, save_path=None):
        """
        Visualize multiclass segmentation results
        
        Args:
            image: Input image
            true_mask: Ground truth mask
            pred_mask: Predicted mask
            class_names: List of class names
            save_path: Path to save visualization
        """
        if class_names is None:
            class_names = self.class_names
        
        # Create color map for classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Input image
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(true_mask, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(image)
        axes[3].imshow(pred_mask, cmap='tab10', vmin=0, vmax=len(class_names)-1, alpha=0.5)
        axes[3].set_title('Prediction Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_multiclass_dataset_from_binary(binary_images, binary_masks, tumor_types):
    """
    Create multiclass dataset from binary segmentation data
    
    Args:
        binary_images: List of binary images
        binary_masks: List of binary masks
        tumor_types: List of tumor type labels for each image
        
    Returns:
        Multiclass images and masks
    """
    # Create mapping from tumor type to class index
    tumor_type_to_class = {tumor_type: idx+1 for idx, tumor_type in enumerate(set(tumor_types))}
    
    # Create multiclass masks
    multiclass_masks = []
    for mask, tumor_type in zip(binary_masks, tumor_types):
        # Start with background (class 0)
        multiclass_mask = np.zeros_like(mask, dtype=np.int32)
        
        # Set tumor region to appropriate class
        class_idx = tumor_type_to_class.get(tumor_type, 0)
        multiclass_mask[mask > 0.5] = class_idx
        
        multiclass_masks.append(multiclass_mask)
    
    return np.array(binary_images), np.array(multiclass_masks)