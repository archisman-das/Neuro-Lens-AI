import argparse
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from src.data import get_datasets, prepare_dataset
from src.models import get_model
from src.utils import make_gradcam_heatmap, overlay_heatmap


def parse_args():
    parser = argparse.ArgumentParser(description='Generate explainability outputs for brain tumor models')
    parser.add_argument('--model', choices=['cnn', 'transfer', 'vit'], default='cnn')
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--examples', type=int, default=4)
    return parser.parse_args()


def _get_default_conv_layer(model_type):
    if model_type == 'cnn':
        return 'conv_block_3'
    if model_type == 'transfer':
        return 'conv5_block3_out'
    return None


def _get_sample_images(dataset, max_examples=4):
    images = []
    labels = []
    for batch, (x, y) in enumerate(dataset):
        for i in range(x.shape[0]):
            if len(images) >= max_examples:
                return np.array(images), np.array(labels)
            images.append(x[i].numpy())
            labels.append(int(y[i].numpy()))
        if len(images) >= max_examples:
            break
    return np.array(images), np.array(labels)


def _plot_image(image, title, save_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(image.astype('uint8'))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def vit_patch_saliency(model, image, image_size=(224, 224)):
    """Saliency for the hybrid ResNet50+ViT classifier defined in src/models.py.

    The hybrid model projects the ResNet50 feature map (7x7) into patch tokens via
    a 1x1 Conv2D named 'hybrid_patch_projection', then reshapes to a sequence and
    passes it through transformer blocks. We compute gradient-based saliency on
    the patch token sequence (after position embedding) and reshape it back to
    a 7x7 grid before resizing to the input resolution.
    """
    try:
        token_layer = model.get_layer('hybrid_patch_tokens')
    except ValueError:
        token_layer = model.get_layer('hybrid_patch_projection')

    token_model = tf.keras.Model(inputs=model.inputs, outputs=token_layer.output)
    image_batch = tf.expand_dims(image, axis=0)
    with tf.GradientTape() as tape:
        tokens = token_model(image_batch)
        tape.watch(tokens)
        predictions = model(image_batch)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, tokens)
    importance = tf.reduce_mean(tf.abs(grads * tokens), axis=-1)
    importance = tf.squeeze(importance).numpy()

    num_tokens = importance.shape[0] if importance.ndim == 1 else importance.size
    side = int(round(num_tokens ** 0.5))
    if side * side != num_tokens:
        side = max(1, int(np.floor(num_tokens ** 0.5)))
        importance = importance[: side * side]
    importance = importance.reshape(side, side)
    importance = importance / (importance.max() + 1e-8)
    importance = tf.expand_dims(importance, axis=-1)
    importance = tf.image.resize(importance, image_size, method='bilinear').numpy()
    return np.squeeze(importance)


def explain_examples(model, model_type, images, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    conv_layer = _get_default_conv_layer(model_type)
    for idx, (image, label) in enumerate(zip(images, labels)):
        title = f'Label={label}'
        sample_image = image.astype('uint8') if image.dtype != 'uint8' else image
        _plot_image(sample_image, f'Input {idx} ({title})', os.path.join(output_dir, f'input_{idx}.png'))

        if model_type in ['cnn', 'transfer']:
            heatmap = make_gradcam_heatmap(tf.expand_dims(image, axis=0), model, conv_layer)
            overlay = overlay_heatmap(sample_image, heatmap)
            _plot_image(overlay, f'Grad-CAM {idx} ({title})', os.path.join(output_dir, f'gradcam_{idx}.png'))

        if model_type == 'vit':
            heatmap = vit_patch_saliency(model, image)
            overlay = overlay_heatmap(sample_image, heatmap)
            _plot_image(overlay, f'ViT Patch Saliency {idx} ({title})', os.path.join(output_dir, f'vit_saliency_{idx}.png'))

        prediction = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0][0]
        print(f'Example {idx}: true={label}, score={prediction:.4f}')


def main():
    args = parse_args()
    train_ds, val_ds, test_ds = get_datasets(args.dataset, batch_size=args.batch_size)
    eval_ds = test_ds if test_ds is not None else val_ds
    if eval_ds is None:
        raise ValueError('No validation or test dataset available for explanation.')

    eval_ds = prepare_dataset(eval_ds)
    model = get_model(args.model)
    model.load_weights(args.weights)

    images, labels = _get_sample_images(eval_ds, max_examples=args.examples)
    explain_dir = os.path.join(args.output, args.model, 'explain')
    explain_examples(model, args.model, images, labels, explain_dir)
    print(f'Explainability outputs saved to {explain_dir}')


if __name__ == '__main__':
    main()
