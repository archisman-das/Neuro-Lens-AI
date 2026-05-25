"""
U-Net Segmentation Models with Attention Mechanisms for Brain Tumor Detection
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


def convolution_block(input_tensor, num_filters, kernel_size=3, dropout_rate=0.2, use_batchnorm=True):
    """
    Double convolution block for U-Net
    
    Args:
        input_tensor: Input tensor
        num_filters: Number of filters in convolution layers
        kernel_size: Kernel size for convolutions
        dropout_rate: Dropout rate for regularization
        use_batchnorm: Whether to use batch normalization
    
    Returns:
        Output tensor after two convolution operations
    """
    x = layers.Conv2D(
        num_filters, 
        kernel_size, 
        activation='relu', 
        padding='same',
        kernel_initializer='he_normal'
    )(input_tensor)
    
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(
        num_filters, 
        kernel_size, 
        activation='relu', 
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    
    return x


def attention_gate(input_tensor, gating_tensor, num_filters):
    """
    Attention gate mechanism for U-Net skip connections
    
    Args:
        input_tensor: Feature map from encoder (skip connection)
        gating_tensor: Gating signal from decoder (upsampled path)
        num_filters: Number of filters
    
    Returns:
        Attended feature map
    """
    # Interpolate gating tensor to match input_tensor spatial dimensions
    gating_tensor = layers.Resizing(
        height=input_tensor.shape[1],
        width=input_tensor.shape[2],
        interpolation='bilinear'
    )(gating_tensor)
    
    # Attention mechanism
    x1 = layers.Conv2D(num_filters, 1, padding='same', use_bias=False)(input_tensor)
    x2 = layers.Conv2D(num_filters, 1, padding='same', use_bias=False)(gating_tensor)
    
    x = layers.Add()([x1, x2])
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
    x = layers.Activation('sigmoid')(x)
    
    # Apply attention weights
    attended_tensor = layers.Multiply()([input_tensor, x])
    
    return attended_tensor


def encoder_block(input_tensor, num_filters, dropout_rate=0.2, use_batchnorm=True):
    """
    Encoder block: convolution + max pooling
    
    Args:
        input_tensor: Input tensor
        num_filters: Number of filters
        dropout_rate: Dropout rate
        use_batchnorm: Whether to use batch normalization
    
    Returns:
        Encoded tensor and skip connection
    """
    x = convolution_block(input_tensor, num_filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p


def decoder_block(input_tensor, skip_tensor, num_filters, use_attention=False, dropout_rate=0.2, use_batchnorm=True):
    """
    Decoder block: upsampling + concatenation with skip connection + convolution
    
    Args:
        input_tensor: Input tensor from previous decoder block
        skip_tensor: Skip connection from encoder
        num_filters: Number of filters
        use_attention: Whether to use attention gate on skip connection
        dropout_rate: Dropout rate
        use_batchnorm: Whether to use batch normalization
    
    Returns:
        Decoded tensor
    """
    # Upsampling
    x = layers.Conv2DTranspose(
        num_filters, 
        (2, 2), 
        strides=(2, 2), 
        padding='same'
    )(input_tensor)
    
    # Apply attention gate if enabled
    if use_attention:
        skip_tensor = attention_gate(skip_tensor, x, num_filters)
    
    # Concatenate with skip connection
    x = layers.Concatenate()([x, skip_tensor])
    
    # Convolution block
    x = convolution_block(x, num_filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    
    return x


def build_unet(
    input_shape=(224, 224, 3),
    num_classes=1,
    base_filters=64,
    dropout_rate=0.2,
    use_batchnorm=True,
    use_attention=False
):
    """
    Build U-Net model for brain tumor segmentation
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (1 for binary segmentation)
        base_filters: Number of filters in first encoder block
        dropout_rate: Dropout rate for regularization
        use_batchnorm: Whether to use batch normalization
        use_attention: Whether to use attention gates in skip connections
    
    Returns:
        U-Net model
    """
    inputs = layers.Input(shape=input_shape, name='image_input')
    
    # Normalize input
    x = layers.Rescaling(1.0 / 255)(inputs)
    
    # Encoder (downsampling path)
    filters = base_filters
    skip_connections = []
    
    for i in range(4):
        s, x = encoder_block(x, filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
        skip_connections.append(s)
        filters *= 2
    
    # Bottleneck
    x = convolution_block(x, filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    
    # Decoder (upsampling path)
    filters //= 2
    
    for i in range(4):
        skip = skip_connections.pop()
        x = decoder_block(
            x, 
            skip, 
            filters, 
            use_attention=use_attention,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm
        )
        filters //= 2
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', padding='same')(x)
    else:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs], name='unet')
    
    return model


def build_attention_unet(input_shape=(224, 224, 3), num_classes=1, base_filters=64, dropout_rate=0.2):
    """
    Build Attention U-Net model (U-Net with attention gates)
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        base_filters: Number of filters in first encoder block
        dropout_rate: Dropout rate
    
    Returns:
        Attention U-Net model
    """
    return build_unet(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=base_filters,
        dropout_rate=dropout_rate,
        use_attention=True
    )


def build_res_unet(
    input_shape=(224, 224, 3),
    num_classes=1,
    base_filters=64,
    dropout_rate=0.2
):
    """
    Build Residual U-Net model (U-Net with residual connections)
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        base_filters: Number of filters in first encoder block
        dropout_rate: Dropout rate
    
    Returns:
        Residual U-Net model
    """
    inputs = layers.Input(shape=input_shape, name='image_input')
    
    # Normalize input
    x = layers.Rescaling(1.0 / 255)(inputs)
    
    # Initial convolution
    x = layers.Conv2D(base_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    
    # Encoder with residual connections
    filters = base_filters
    skip_connections = []
    
    for i in range(4):
        residual = layers.Conv2D(
            filters, 1, padding='same', kernel_initializer='he_normal'
        )(x)
        
        x = convolution_block(x, filters, dropout_rate=dropout_rate)
        
        # Add residual connection
        x = layers.Add()([x, residual])
        
        skip_connections.append(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        filters *= 2
    
    # Bottleneck
    x = convolution_block(x, filters, dropout_rate=dropout_rate)
    
    # Decoder
    filters //= 2
    
    for i in range(4):
        x = layers.Conv2DTranspose(
            filters, 
            (2, 2), 
            strides=(2, 2), 
            padding='same'
        )(x)
        
        skip = skip_connections.pop()
        x = layers.Concatenate()([x, skip])
        x = convolution_block(x, filters, dropout_rate=dropout_rate)
        
        filters //= 2
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', padding='same')(x)
    else:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs], name='res_unet')
    
    return model


def build_multi_modal_unet(
    input_shapes=[(224, 224, 3), (224, 224, 3)],
    num_classes=1,
    base_filters=64,
    dropout_rate=0.2,
    fusion_method='attention'
):
    """
    Build multi-modal U-Net with attention fusion for combining multiple input modalities
    
    Args:
        input_shapes: List of input shapes for different modalities
        num_classes: Number of output classes
        base_filters: Number of filters in first encoder block
        dropout_rate: Dropout rate
        fusion_method: Method for fusing modalities ('attention', 'concat', 'add')
    
    Returns:
        Multi-modal U-Net model
    """
    inputs = [layers.Input(shape=shape, name=f'modality_{i}_input') for i, shape in enumerate(input_shapes)]
    
    # Process each modality through initial convolutions
    modality_features = []
    for i, inp in enumerate(inputs):
        x = layers.Rescaling(1.0 / 255)(inp)
        x = layers.Conv2D(base_filters, 3, padding='same', kernel_initializer='he_normal')(x)
        modality_features.append(x)
    
    # Fuse modalities
    if fusion_method == 'attention':
        # Attention-based fusion
        fused = attention_fusion(modality_features, base_filters)
    elif fusion_method == 'concat':
        fused = layers.Concatenate()(modality_features)
        fused = layers.Conv2D(base_filters, 1, padding='same')(fused)
    elif fusion_method == 'add':
        fused = layers.Add()(modality_features)
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Continue with U-Net architecture
    filters = base_filters
    skip_connections = []
    
    # Encoder
    for i in range(4):
        s, fused = encoder_block(fused, filters, dropout_rate=dropout_rate)
        skip_connections.append(s)
        filters *= 2
    
    # Bottleneck
    fused = convolution_block(fused, filters, dropout_rate=dropout_rate)
    
    # Decoder
    filters //= 2
    
    for i in range(4):
        skip = skip_connections.pop()
        fused = decoder_block(fused, skip, filters, dropout_rate=dropout_rate)
        filters //= 2
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', padding='same')(fused)
    else:
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(fused)
    
    model = Model(inputs=inputs, outputs=outputs, name='multi_modal_unet')
    
    return model


def attention_fusion(feature_maps, num_filters):
    """
    Attention-based fusion of multiple feature maps
    
    Args:
        feature_maps: List of feature maps to fuse
        num_filters: Number of filters for attention computation
    
    Returns:
        Fused feature map
    """
    if len(feature_maps) == 1:
        return feature_maps[0]
    
    # Compute attention weights for each feature map
    attention_weights = []
    for fm in feature_maps:
        # Global average pooling to get channel-wise statistics
        gap = layers.GlobalAveragePooling2D()(fm)
        # Learn attention weights
        w = layers.Dense(num_filters, activation='relu')(gap)
        w = layers.Dense(num_filters, activation='softmax')(w)
        w = layers.Reshape((1, 1, num_filters))(w)
        attention_weights.append(w)
    
    # Apply attention weights and sum
    weighted_features = [layers.Multiply()([fm, w]) for fm, w in zip(feature_maps, attention_weights)]
    fused = layers.Add()(weighted_features)
    
    return fused


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for segmentation evaluation
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient value
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss function for training segmentation models
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor
    
    Returns:
        Dice loss value
    """
    return 1 - dice_coefficient(y_true, y_pred, smooth)


def combined_loss(weights=None):
    """
    Combined loss function (weighted sum of dice loss and binary crossentropy)
    
    Args:
        weights: Weights for each loss component [dice_weight, bce_weight]
    
    Returns:
        Combined loss function
    """
    if weights is None:
        weights = [0.5, 0.5]
    
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        return weights[0] * dice + weights[1] * bce
    
    return loss


def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU) metric for segmentation
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor
    
    Returns:
        IoU value
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def get_segmentation_model(model_name='unet', **kwargs):
    """
    Factory function to get segmentation models
    
    Args:
        model_name: Name of the model ('unet', 'attention_unet', 'res_unet', 'multi_modal_unet')
        **kwargs: Additional arguments passed to the model builder
    
    Returns:
        Segmentation model
    """
    models = {
        'unet': build_unet,
        'attention_unet': build_attention_unet,
        'res_unet': build_res_unet,
        'multi_modal_unet': build_multi_modal_unet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)