"""
Advanced Models: 3D MRI Transformer, Federated Learning, and Self-Supervised Pre-training
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import json
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
from datetime import datetime


class MRI3DTransformer:
    """
    3D Vision Transformer for MRI volume analysis
    """
    
    def __init__(
        self,
        input_shape=(128, 128, 128, 1),
        patch_size=16,
        num_layers=12,
        num_heads=12,
        embedding_dim=768,
        mlp_dim=3072,
        dropout_rate=0.1,
        num_classes=1000
    ):
        """
        Initialize 3D MRI Transformer
        
        Args:
            input_shape: Shape of input 3D volumes (depth, height, width, channels)
            patch_size: Size of patches to extract
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            embedding_dim: Dimension of patch embeddings
            mlp_dim: Dimension of MLP hidden layer
            dropout_rate: Dropout rate
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.model = None
    
    def patch_embedding(self, inputs):
        """
        Extract patches and create embeddings
        
        Args:
            inputs: Input 3D volume
            
        Returns:
            Patch embeddings
        """
        # Calculate number of patches
        depth_patches = self.input_shape[0] // self.patch_size
        height_patches = self.input_shape[1] // self.patch_size
        width_patches = self.input_shape[2] // self.patch_size
        num_patches = depth_patches * height_patches * width_patches
        
        # Extract patches
        patches = layers.Reshape((num_patches, self.patch_size ** 3 * self.input_shape[3]))(inputs)
        
        # Project patches to embedding dimension
        patch_projection = layers.Dense(self.embedding_dim, name='patch_projection')(patches)
        
        return patch_projection, num_patches
    
    def transformer_encoder(self, x):
        """
        Build transformer encoder layers
        
        Args:
            x: Input embeddings
            
        Returns:
            Encoded representations
        """
        for i in range(self.num_layers):
            # Layer normalization
            x_norm = layers.LayerNormalization(epsilon=1e-6, name=f'layer_norm_{i}')(x)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embedding_dim // self.num_heads,
                dropout=self.dropout_rate,
                name=f'attention_{i}'
            )(x_norm, x_norm)
            
            # Add & norm
            x = layers.Add(name=f'add_{i}')([x, attention_output])
            
            # MLP block
            x_norm = layers.LayerNormalization(epsilon=1e-6, name=f'layer_norm_{i}_mlp')(x)
            mlp_output = layers.Dense(self.mlp_dim, activation='gelu', name=f'mlp_dense1_{i}')(x_norm)
            mlp_output = layers.Dropout(self.dropout_rate, name=f'mlp_dropout1_{i}')(mlp_output)
            mlp_output = layers.Dense(self.embedding_dim, name=f'mlp_dense2_{i}')(mlp_output)
            mlp_output = layers.Dropout(self.dropout_rate, name=f'mlp_dropout2_{i}')(mlp_output)
            
            # Add & norm
            x = layers.Add(name=f'add_mlp_{i}')([x, mlp_output])
        
        return x
    
    def build_model(self):
        """
        Build the 3D MRI Transformer model
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape, name='mri_volume')
        
        # Patch embedding
        patches, num_patches = self.patch_embedding(inputs)
        
        # Positional encoding
        pos_encoding = self.get_positional_encoding(num_patches, self.embedding_dim)
        x = layers.Add()([patches, pos_encoding])
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Classification head
        x = layers.Dense(self.embedding_dim, activation='tanh')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='mri_3d_transformer')
        
        return self.model
    
    def get_positional_encoding(self, num_patches, embedding_dim):
        """
        Generate positional encoding
        
        Args:
            num_patches: Number of patches
            embedding_dim: Embedding dimension
            
        Returns:
            Positional encoding tensor
        """
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=embedding_dim,
            name='positional_encoding'
        )(positions)
        
        return position_embedding
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, callbacks=None):
        """
        Train the model
        
        Args:
            X_train: Training volumes
            y_train: Training labels
            X_val: Validation volumes
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.compile_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = []
        
        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ])
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, save_path):
        """Save model"""
        if self.model is not None:
            self.model.save(save_path)
    
    def load_model(self, model_path):
        """Load model"""
        self.model = tf.keras.models.load_model(model_path)
        return self.model


class FederatedLearningClient:
    """
    Client for federated learning
    """
    
    def __init__(self, client_id, model, data, batch_size=32):
        """
        Initialize federated learning client
        
        Args:
            client_id: Unique client identifier
            model: Model architecture (uncompiled)
            data: Tuple of (X, y) for this client
            batch_size: Batch size for training
        """
        self.client_id = client_id
        self.model = model
        self.X, self.y = data
        self.batch_size = batch_size
        self.local_epochs = 1
    
    def set_weights(self, weights):
        """Set model weights from server"""
        self.model.set_weights(weights)
    
    def get_weights(self):
        """Get model weights to send to server"""
        return self.model.get_weights()
    
    def train_local(self, epochs=1, batch_size=32):
        """
        Train model locally
        
        Args:
            epochs: Number of local training epochs
            batch_size: Batch size
            
        Returns:
            Updated weights
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            self.X, self.y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        return self.get_weights(), history.history


class FederatedLearningServer:
    """
    Server for federated learning coordination
    """
    
    def __init__(self, model_architecture, input_shape, num_classes):
        """
        Initialize federated learning server
        
        Args:
            model_architecture: Function that creates model architecture
            input_shape: Shape of input data
            num_classes: Number of output classes
        """
        self.model_architecture = model_architecture
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.global_model = self.create_model()
        self.clients = []
        self.round_history = []
    
    def create_model(self):
        """Create global model"""
        model = self.model_architecture(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def add_client(self, client):
        """Add client to federation"""
        self.clients.append(client)
    
    def aggregate_weights(self, client_weights, client_data_sizes):
        """
        Aggregate client weights using FedAvg
        
        Args:
            client_weights: List of client weight sets
            client_data_sizes: List of client data sizes
            
        Returns:
            Aggregated weights
        """
        total_size = sum(client_data_sizes)
        aggregated_weights = []
        
        # For each layer
        for layer_weights in zip(*client_weights):
            # Weighted average
            aggregated = np.zeros_like(layer_weights[0])
            for weight, size in zip(layer_weights, client_data_sizes):
                aggregated += weight * (size / total_size)
            aggregated_weights.append(aggregated)
        
        return aggregated_weights
    
    def run_federation_round(self, local_epochs=1):
        """
        Run one round of federated learning
        
        Args:
            local_epochs: Number of local training epochs
            
        Returns:
            Dictionary with round metrics
        """
        # Get current global weights
        global_weights = self.global_model.get_weights()
        
        # Send weights to clients and train locally
        client_weights = []
        client_data_sizes = []
        
        for client in self.clients:
            # Set client weights
            client.set_weights(global_weights)
            
            # Train locally
            weights, history = client.train_local(epochs=local_epochs)
            
            client_weights.append(weights)
            client_data_sizes.append(len(client.X))
        
        # Aggregate weights
        aggregated_weights = self.aggregate_weights(client_weights, client_data_sizes)
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        # Evaluate global model on each client
        client_accuracies = []
        for client in self.clients:
            _, accuracy = self.global_model.evaluate(client.X, client.y, verbose=0)
            client_accuracies.append(accuracy)
        
        # Store round metrics
        round_metrics = {
            'round': len(self.round_history) + 1,
            'client_accuracies': client_accuracies,
            'mean_accuracy': np.mean(client_accuracies),
            'std_accuracy': np.std(client_accuracies)
        }
        
        self.round_history.append(round_metrics)
        
        return round_metrics
    
    def run_federation(self, num_rounds=10, local_epochs=1):
        """
        Run multiple rounds of federated learning
        
        Args:
            num_rounds: Number of federation rounds
            local_epochs: Number of local training epochs per round
            
        Returns:
            List of round metrics
        """
        for round_num in range(num_rounds):
            print(f"\nRound {round_num + 1}/{num_rounds}")
            metrics = self.run_federation_round(local_epochs)
            print(f"Mean accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        
        return self.round_history
    
    def save_results(self, save_dir='./federated_results'):
        """Save federation results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save round history
        with open(os.path.join(save_dir, 'federation_history.json'), 'w') as f:
            json.dump(self.round_history, f, indent=2)
        
        # Save global model
        self.global_model.save(os.path.join(save_dir, 'global_model.h5'))
        
        # Save client configurations
        client_configs = []
        for client in self.clients:
            client_configs.append({
                'client_id': client.client_id,
                'data_size': len(client.X)
            })
        
        with open(os.path.join(save_dir, 'client_configs.json'), 'w') as f:
            json.dump(client_configs, f, indent=2)


class SelfSupervisedPretrainer:
    """
    Self-supervised pre-training for medical images
    """
    
    def __init__(self, model_architecture, input_shape, projection_dim=128):
        """
        Initialize self-supervised pretrainer
        
        Args:
            model_architecture: Function that creates encoder architecture
            input_shape: Shape of input images
            projection_dim: Dimension of projection head output
        """
        self.model_architecture = model_architecture
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.encoder = None
        self.pretext_model = None
    
    def create_encoder(self):
        """Create encoder model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Base encoder (e.g., ResNet, U-Net encoder)
        x = self.model_architecture(inputs)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Projection head
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.projection_dim)(x)
        
        self.encoder = Model(inputs=inputs, outputs=x, name='encoder')
        
        return self.encoder
    
    def create_pretext_model(self, pretext_task='rotation'):
        """
        Create pretext task model
        
        Args:
            pretext_task: Type of pretext task ('rotation', 'jigsaw', 'contrastive')
            
        Returns:
            Pretext task model
        """
        if self.encoder is None:
            self.create_encoder()
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        x = self.encoder(inputs)
        
        # Pretext task head
        if pretext_task == 'rotation':
            # 4-class rotation prediction (0°, 90°, 180°, 270°)
            outputs = layers.Dense(4, activation='softmax', name='rotation_head')(x)
        elif pretext_task == 'jigsaw':
            # Jigsaw puzzle solving (number of permutations)
            num_permutations = 10  # Example
            outputs = layers.Dense(num_permutations, activation='softmax', name='jigsaw_head')(x)
        elif pretext_task == 'contrastive':
            # Contrastive learning (similarity score)
            outputs = layers.Dense(1, activation='sigmoid', name='contrastive_head')(x)
        else:
            raise ValueError(f"Unknown pretext task: {pretext_task}")
        
        self.pretext_model = Model(inputs=inputs, outputs=outputs, name=f'pretext_{pretext_task}')
        
        return self.pretext_model
    
    def prepare_rotation_data(self, X):
        """
        Prepare data for rotation prediction pretext task
        
        Args:
            X: Input images
            
        Returns:
            Rotated images and rotation labels
        """
        rotated_images = []
        rotation_labels = []
        
        for image in X:
            # Generate 4 rotated versions
            for rotation_angle in [0, 90, 180, 270]:
                rotated = np.rot90(image, k=rotation_angle // 90, axes=(0, 1))
                rotated_images.append(rotated)
                rotation_labels.append(rotation_angle // 90)
        
        return np.array(rotated_images), np.array(rotation_labels)
    
    def prepare_jigsaw_data(self, X, grid_size=3):
        """
        Prepare data for jigsaw puzzle pretext task
        
        Args:
            X: Input images
            grid_size: Size of jigsaw grid
            
        Returns:
            Permuted images and permutation labels
        """
        # Simple implementation: just shuffle patches
        permuted_images = []
        permutation_labels = []
        
        for image in X:
            # Divide image into patches
            h, w = image.shape[0] // grid_size, image.shape[1] // grid_size
            patches = []
            for i in range(grid_size):
                for j in range(grid_size):
                    patch = image[i*h:(i+1)*h, j*w:(j+1)*w, :]
                    patches.append(patch)
            
            # Create a few random permutations
            for _ in range(5):  # 5 random permutations per image
                perm = np.random.permutation(len(patches))
                permuted = np.concatenate([
                    np.concatenate([patches[perm[i*grid_size + j]] for j in range(grid_size)], axis=1)
                    for i in range(grid_size)
                ], axis=0)
                permuted_images.append(permuted)
                permutation_labels.append(hash(tuple(perm)) % 10)  # Simplified
        
        return np.array(permuted_images), np.array(permutation_labels)
    
    def pretrain(self, X, pretext_task='rotation', epochs=50, batch_size=32):
        """
        Perform self-supervised pre-training
        
        Args:
            X: Input images
            pretext_task: Type of pretext task
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        # Create pretext model
        self.create_pretext_model(pretext_task)
        
        # Prepare pretext data
        if pretext_task == 'rotation':
            X_pretext, y_pretext = self.prepare_rotation_data(X)
        elif pretext_task == 'jigsaw':
            X_pretext, y_pretext = self.prepare_jigsaw_data(X)
        else:
            X_pretext, y_pretext = X, np.zeros(len(X))  # Placeholder
        
        # Compile pretext model
        self.pretext_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = self.pretext_model.fit(
            X_pretext, y_pretext,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def get_pretrained_encoder(self):
        """Get the pretrained encoder"""
        if self.encoder is None:
            raise ValueError("Encoder not created. Call pretrain() first.")
        return self.encoder
    
    def save_pretrained_encoder(self, save_path):
        """Save pretrained encoder"""
        if self.encoder is not None:
            self.encoder.save(save_path)
    
    def load_pretrained_encoder(self, model_path):
        """Load pretrained encoder"""
        self.encoder = tf.keras.models.load_model(model_path)
        return self.encoder


class MedicalImageDataset:
    """
    Dataset class for medical images with support for various modalities
    """
    
    def __init__(self, data_dir, modality='mri', image_size=(224, 224)):
        """
        Initialize medical image dataset
        
        Args:
            data_dir: Directory containing medical images
            modality: Imaging modality ('mri', 'ct', 'xray')
            image_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.modality = modality
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.metadata = []
    
    def load_data(self):
        """Load medical images from directory"""
        # Placeholder - implement based on your data structure
        pass
    
    def augment(self, image, label):
        """Apply medical image-specific augmentations"""
        # Random rotations (medical images often have consistent orientation)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            image = tf.image.rot90(image, k=np.random.randint(0, 4))
        
        # Random flips (only if anatomically appropriate)
        if np.random.random() > 0.5:
            image = tf.image.flip_left_right(image)
        
        # Random brightness/contrast adjustments
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        return image, label
    
    def create_dataset(self, batch_size=32, shuffle=True, augment=False):
        """Create TensorFlow dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.images))
        
        if augment:
            dataset = dataset.map(
                lambda x, y: self.augment(x, y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset


def create_medical_vision_transformer(input_shape=(224, 224, 3), num_classes=1000, patch_size=16):
    """
    Create a Vision Transformer for 2D medical images
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        patch_size: Size of patches
        
    Returns:
        ViT model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Patch extraction and embedding
    x = layers.Rescaling(1.0 / 255)(inputs)
    patches = layers.Conv2D(
        filters=64,
        kernel_size=patch_size,
        strides=patch_size,
        padding='same',
        activation='relu'
    )(x)
    
    # Reshape to sequence
    patch_shape = patches.shape[1:]
    patches = layers.Reshape((patch_shape[0] * patch_shape[1], patch_shape[2]))(patches)
    
    # Positional encoding
    num_patches = patch_shape[0] * patch_shape[1]
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=64)(positions)
    patches = layers.Add()([patches, position_embedding])
    
    # Transformer blocks
    for i in range(6):
        # Multi-head attention
        x = layers.LayerNormalization()(patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=64
        )(x, x)
        x = layers.Add()([x, attention_output])
        
        # MLP
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64)(x)
        patches = layers.Add()([x, patches])
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(patches)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='medical_vit')


def create_self_supervised_model(input_shape=(224, 224, 3), projection_dim=128):
    """
    Create a model for self-supervised learning (SimCLR-style)
    
    Args:
        input_shape: Input image shape
        projection_dim: Dimension of projection head
        
    Returns:
        Base encoder model and projection model
    """
    # Base encoder (ResNet-like)
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    
    # Simple CNN encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Representation (for downstream tasks)
    representation = layers.Dense(256, activation='relu', name='representation')(x)
    
    # Projection head (for contrastive loss)
    projection = layers.Dense(projection_dim, name='projection')(representation)
    
    # Create models
    encoder = Model(inputs=inputs, outputs=representation, name='encoder')
    projection_model = Model(inputs=inputs, outputs=projection, name='projection_model')
    
    return encoder, projection_model