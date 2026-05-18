import tensorflow as tf
from tensorflow.keras import layers


def build_cnn_baseline(input_shape=(224, 224, 3), dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape, name='image_input')
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv_block_1')(x)
    x = layers.MaxPooling2D(name='pool_block_1')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv_block_2')(x)
    x = layers.MaxPooling2D(name='pool_block_2')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv_block_3')(x)
    x = layers.MaxPooling2D(name='pool_block_3')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    return tf.keras.Model(inputs, outputs, name='cnn_baseline')


def build_transfer_model(
    input_shape=(224, 224, 3),
    dropout_rate=0.3,
    base_model_name='resnet50',
    weights='imagenet',
    fine_tune=False,
    fine_tune_at=None,
):
    inputs = tf.keras.Input(shape=input_shape, name='image_input')

    if base_model_name.lower() == 'resnet50':
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling='avg',
            name='resnet_base',
        )
    elif base_model_name.lower() == 'vgg16':
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling='avg',
            name='vgg_base',
        )
    else:
        raise ValueError('Unsupported base_model_name. Use resnet50 or vgg16.')

    base_model.trainable = fine_tune
    if fine_tune and fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    return tf.keras.Model(inputs, outputs, name='transfer_model')


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def transformer_block(x, num_heads, projection_dim, mlp_dim, dropout_rate, block_id=None):
    x1 = layers.LayerNormalization(epsilon=1e-6, name=f'vit_norm_{block_id}_1')(x)
    attention_layer = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=dropout_rate,
        name=f'vit_attention_{block_id}',
    )
    attention_output = attention_layer(x1, x1)
    x2 = layers.Add()([attention_output, x])
    x3 = layers.LayerNormalization(epsilon=1e-6, name=f'vit_norm_{block_id}_2')(x2)
    x3 = layers.Dense(mlp_dim, activation='gelu')(x3)
    x3 = layers.Dropout(dropout_rate)(x3)
    x3 = layers.Dense(projection_dim)(x3)
    x3 = layers.Dropout(dropout_rate)(x3)
    x4 = layers.Add()([x3, x2])
    return x4


def build_vit_classifier(
    input_shape=(224, 224, 3),
    patch_size=16,
    num_layers=4,
    num_heads=4,
    projection_dim=128,
    mlp_dim=256,
    dropout_rate=0.1,
    weights='imagenet',
):
    inputs = tf.keras.Input(shape=input_shape, name='image_input')
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None,
        name='vit_hybrid_resnet_base',
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.Conv2D(projection_dim, 1, padding='same', name='hybrid_patch_projection')(x)
    num_patches = (input_shape[0] // 32) * (input_shape[1] // 32)
    patches = layers.Reshape((num_patches, projection_dim), name='hybrid_patch_tokens')(x)
    x = PatchEncoder(num_patches, projection_dim)(patches)

    for i in range(num_layers):
        x = transformer_block(x, num_heads, projection_dim, mlp_dim, dropout_rate, block_id=i)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vit_classifier')


def get_model(
    model_name,
    input_shape=(224, 224, 3),
    transfer_weights='imagenet',
    fine_tune_transfer=False,
    transfer_fine_tune_at=None,
):
    model_name = model_name.lower()
    if model_name == 'cnn':
        return build_cnn_baseline(input_shape=input_shape)
    if model_name == 'transfer':
        return build_transfer_model(
            input_shape=input_shape,
            weights=transfer_weights,
            fine_tune=fine_tune_transfer,
            fine_tune_at=transfer_fine_tune_at,
        )
    if model_name == 'vit':
        return build_vit_classifier(input_shape=input_shape, weights=transfer_weights)
    raise ValueError('Unknown model_name. Use cnn, transfer, or vit.')
