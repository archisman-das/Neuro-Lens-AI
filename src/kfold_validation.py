"""
K-Fold Cross-Validation Framework for Brain Tumor Detection
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime


class KFoldValidator:
    """
    K-Fold Cross-Validation wrapper for brain tumor detection models
    """
    
    def __init__(
        self,
        model_builder,
        n_splits=5,
        shuffle=True,
        random_state=42,
        stratified=True
    ):
        """
        Initialize K-Fold validator
        
        Args:
            model_builder: Function that builds and returns a compiled model
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            stratified: Whether to use stratified k-fold (for classification)
        """
        self.model_builder = model_builder
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified
        
        # Results storage
        self.fold_histories = []
        self.fold_metrics = []
        self.best_models = []
        
    def split_data(self, X, y=None):
        """
        Split data into k folds
        
        Args:
            X: Feature data (images)
            y: Labels (optional, for stratified splitting)
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if self.stratified and y is not None:
            kf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            return list(kf.split(X, y))
        else:
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            return list(kf.split(X))
    
    def train_fold(
        self,
        fold_idx,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=32,
        callbacks=None,
        **fit_kwargs
    ):
        """
        Train model on a single fold
        
        Args:
            fold_idx: Index of the current fold
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            **fit_kwargs: Additional arguments for model.fit()
            
        Returns:
            Trained model and training history
        """
        # Build and compile model
        model = self.model_builder()
        
        # Default callbacks
        if callbacks is None:
            callbacks = []
        
        # Add early stopping if not provided
        if not any(isinstance(c, tf.keras.callbacks.EarlyStopping) for c in callbacks):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            )
        
        # Add model checkpoint if not provided
        if not any(isinstance(c, tf.keras.callbacks.ModelCheckpoint) for c in callbacks):
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f'best_model_fold_{fold_idx}.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            )
        
        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **fit_kwargs
        )
        
        # Evaluate on validation set
        val_results = model.evaluate(X_val, y_val, verbose=0)
        
        # Store results
        self.fold_histories.append(history)
        self.fold_metrics.append({
            'fold': fold_idx,
            'val_loss': val_results[0] if isinstance(val_results, list) else val_results,
            'val_metrics': {
                metric_name: float(val_results[i]) 
                for i, metric_name in enumerate(model.metrics_names)
            } if isinstance(val_results, list) else {'loss': float(val_results)},
            'epochs_trained': len(history.history['loss'])
        })
        
        # Store best model
        self.best_models.append(model)
        
        return model, history
    
    def cross_validate(
        self,
        X,
        y=None,
        epochs=50,
        batch_size=32,
        callbacks=None,
        save_dir='./kfold_results',
        **fit_kwargs
    ):
        """
        Perform k-fold cross-validation
        
        Args:
            X: Feature data (images)
            y: Labels (optional, for stratified splitting)
            epochs: Number of training epochs per fold
            batch_size: Batch size
            callbacks: List of Keras callbacks
            save_dir: Directory to save results
            **fit_kwargs: Additional arguments for model.fit()
            
        Returns:
            Dictionary containing cross-validation results
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Split data
        folds = self.split_data(X, y)
        
        # Reset results
        self.fold_histories = []
        self.fold_metrics = []
        self.best_models = []
        
        # Train on each fold
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold_idx + 1}/{self.n_splits}")
            print(f"{'='*50}")
            
            # Split data
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices] if y is not None else (None, None)
            
            # Train fold
            model, history = self.train_fold(
                fold_idx,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                **fit_kwargs
            )
            
            print(f"Fold {fold_idx + 1} completed. Validation loss: {self.fold_metrics[-1]['val_loss']:.4f}")
        
        # Calculate aggregate metrics
        results = self.summarize_results()
        
        # Save results
        self.save_results(results, save_dir)
        
        return results
    
    def summarize_results(self):
        """
        Summarize cross-validation results
        
        Returns:
            Dictionary containing aggregated metrics
        """
        if not self.fold_metrics:
            raise ValueError("No fold metrics found. Run cross_validate first.")
        
        # Extract metrics
        val_losses = [m['val_loss'] for m in self.fold_metrics]
        
        # Get all metric names from first fold
        metric_names = list(self.fold_metrics[0]['val_metrics'].keys())
        metric_values = {name: [] for name in metric_names}
        
        for m in self.fold_metrics:
            for name in metric_names:
                metric_values[name].append(m['val_metrics'].get(name, 0))
        
        # Calculate statistics
        summary = {
            'n_splits': self.n_splits,
            'fold_results': self.fold_metrics,
            'aggregate_metrics': {
                'val_loss': {
                    'mean': float(np.mean(val_losses)),
                    'std': float(np.std(val_losses)),
                    'min': float(np.min(val_losses)),
                    'max': float(np.max(val_losses))
                }
            }
        }
        
        # Add metrics statistics
        for name in metric_names:
            values = metric_values[name]
            summary['aggregate_metrics'][name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return summary
    
    def save_results(self, results, save_dir):
        """
        Save cross-validation results to files
        
        Args:
            results: Results dictionary from summarize_results()
            save_dir: Directory to save results
        """
        # Save summary as JSON
        summary_path = os.path.join(save_dir, 'kfold_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed metrics as CSV
        metrics_df = pd.DataFrame(self.fold_metrics)
        metrics_df.to_csv(os.path.join(save_dir, 'fold_metrics.csv'), index=False)
        
        # Save individual fold histories
        for i, history in enumerate(self.fold_histories):
            history_dict = history.history
            history_df = pd.DataFrame(history_dict)
            history_df.to_csv(os.path.join(save_dir, f'fold_{i}_history.csv'), index=False)
        
        # Save models
        models_dir = os.path.join(save_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for i, model in enumerate(self.best_models):
            model.save(os.path.join(models_dir, f'fold_{i}_model.h5'))
        
        print(f"Results saved to {save_dir}")
    
    def get_ensemble_predictions(self, X_test, threshold=0.5):
        """
        Get ensemble predictions from all fold models
        
        Args:
            X_test: Test data
            threshold: Classification threshold (for binary classification)
            
        Returns:
            Ensemble predictions (probabilities and binary predictions)
        """
        if not self.best_models:
            raise ValueError("No models found. Run cross_validate first.")
        
        # Get predictions from each model
        predictions = []
        for model in self.best_models:
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Average predictions
        avg_predictions = np.mean(predictions, axis=0)
        
        # Binary predictions
        binary_predictions = (avg_predictions >= threshold).astype(int)
        
        return avg_predictions, binary_predictions


class SegmentationKFoldValidator(KFoldValidator):
    """
    K-Fold Cross-Validation for segmentation models
    """
    
    def __init__(
        self,
        model_builder,
        n_splits=5,
        shuffle=True,
        random_state=42,
        image_size=(224, 224)
    ):
        """
        Initialize segmentation K-Fold validator
        
        Args:
            model_builder: Function that builds and returns a compiled segmentation model
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed
            image_size: Size of input images
        """
        super().__init__(
            model_builder=model_builder,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            stratified=False  # Segmentation typically doesn't use stratification
        )
        self.image_size = image_size
    
    def create_segmentation_dataset(
        self,
        images,
        masks,
        batch_size=16,
        augment=False
    ):
        """
        Create TensorFlow dataset for segmentation
        
        Args:
            images: Array of input images
            masks: Array of segmentation masks
            batch_size: Batch size
            augment: Whether to apply data augmentation
            
        Returns:
            TensorFlow dataset
        """
        def generator():
            for img, mask in zip(images, masks):
                yield img, mask
        
        def augment_fn(image, mask):
            # Random flips
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)
            
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_up_down(image)
                mask = tf.image.flip_up_down(mask)
            
            # Random rotation
            k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
            mask = tf.image.rot90(mask, k=k)
            
            return image, mask
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(*self.image_size, 1), dtype=tf.float32)
            )
        )
        
        if augment:
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def cross_validate(
        self,
        images,
        masks,
        epochs=50,
        batch_size=16,
        callbacks=None,
        save_dir='./kfold_segmentation_results',
        augment=True
    ):
        """
        Perform k-fold cross-validation for segmentation
        
        Args:
            images: Array of input images
            masks: Array of segmentation masks
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            save_dir: Directory to save results
            augment: Whether to use data augmentation
            
        Returns:
            Dictionary containing cross-validation results
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Split data
        folds = self.split_data(images)
        
        # Reset results
        self.fold_histories = []
        self.fold_metrics = []
        self.best_models = []
        
        # Train on each fold
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            print(f"\n{'='*50}")
            print(f"Training Segmentation Fold {fold_idx + 1}/{self.n_splits}")
            print(f"{'='*50}")
            
            # Split data
            X_train, X_val = images[train_indices], images[val_indices]
            y_train, y_val = masks[train_indices], masks[val_indices]
            
            # Create datasets
            train_dataset = self.create_segmentation_dataset(
                X_train, y_train, batch_size=batch_size, augment=augment
            )
            val_dataset = self.create_segmentation_dataset(
                X_val, y_val, batch_size=batch_size, augment=False
            )
            
            # Train fold
            model, history = self.train_fold_segmentation(
                fold_idx,
                train_dataset,
                val_dataset,
                epochs=epochs,
                callbacks=callbacks
            )
            
            print(f"Fold {fold_idx + 1} completed.")
        
        # Calculate aggregate metrics
        results = self.summarize_results()
        
        # Save results
        self.save_results(results, save_dir)
        
        return results
    
    def train_fold_segmentation(
        self,
        fold_idx,
        train_dataset,
        val_dataset,
        epochs=50,
        callbacks=None
    ):
        """
        Train segmentation model on a single fold
        
        Args:
            fold_idx: Fold index
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            callbacks: List of callbacks
            
        Returns:
            Trained model and history
        """
        # Build model
        model = self.model_builder()
        
        # Default callbacks
        if callbacks is None:
            callbacks = []
        
        # Add early stopping
        if not any(isinstance(c, tf.keras.callbacks.EarlyStopping) for c in callbacks):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            )
        
        # Add model checkpoint
        if not any(isinstance(c, tf.keras.callbacks.ModelCheckpoint) for c in callbacks):
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f'best_segmentation_model_fold_{fold_idx}.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            )
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Evaluate
        val_results = model.evaluate(val_dataset, verbose=0)
        
        # Store results
        self.fold_histories.append(history)
        self.fold_metrics.append({
            'fold': fold_idx,
            'val_loss': val_results[0] if isinstance(val_results, list) else val_results,
            'val_metrics': {
                metric_name: float(val_results[i]) 
                for i, metric_name in enumerate(model.metrics_names)
            } if isinstance(val_results, list) else {'loss': float(val_results)},
            'epochs_trained': len(history.history['loss'])
        })
        
        self.best_models.append(model)
        
        return model, history


def prepare_data_for_kfold(
    image_paths,
    label_paths=None,
    image_size=(224, 224),
    test_size=0.2,
    random_state=42
):
    """
    Prepare data for k-fold cross-validation
    
    Args:
        image_paths: List of paths to image files
        label_paths: List of paths to label/mask files (for segmentation)
        image_size: Size to resize images to
        test_size: Proportion of data to hold out for final testing
        random_state: Random seed
        
    Returns:
        Arrays of images and labels/masks
    """
    import cv2
    from tqdm import tqdm
    
    # Load images
    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        images.append(img)
    
    images = np.array(images)
    
    # Load labels/masks if provided
    if label_paths is not None:
        labels = []
        for path in tqdm(label_paths, desc="Loading labels"):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, image_size)
            mask = mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1)
            labels.append(mask)
        
        labels = np.array(labels)
        
        # Split into train and test
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                images, labels, test_size=test_size, random_state=random_state
            )
            return X_train, y_train, X_test, y_test
        
        return images, labels
    
    return images