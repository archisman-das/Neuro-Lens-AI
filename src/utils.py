import json
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


def save_history(history, filepath):
    if history is None:
        return
    data = {key: value for key, value in history.history.items()}
    np.savez_compressed(filepath, **data)


def plot_training_history(history, output_dir):
    if history is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def save_metrics_json(metrics, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


def load_metrics_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_history_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key].tolist() for key in data.files}


def compute_metrics(model, dataset):
    y_true = []
    y_pred = []
    y_score = []
    for images, labels in dataset:
        logits = model.predict(images, verbose=0)
        probs = logits.flatten()
        predictions = (probs >= 0.5).astype(int)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())
        y_score.extend(probs.tolist())

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    metrics = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
    }
    return metrics


def _find_layer(model, layer_name):
    try:
        return model.get_layer(layer_name)
    except ValueError:
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    return layer.get_layer(layer_name)
                except ValueError:
                    continue
        raise


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    conv_layer = _find_layer(model, last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4, colormap='viridis'):
    import matplotlib.cm as cm
    image = np.array(image, dtype=np.uint8)
    if heatmap.ndim == 2 and heatmap.shape[:2] != image.shape[:2]:
        heatmap = tf.image.resize(heatmap[..., np.newaxis], image.shape[:2], method='bilinear').numpy().squeeze()
    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap(colormap)
    colored = colormap(heatmap)
    colored = tf.keras.preprocessing.image.array_to_img(colored)
    colored = np.array(colored)
    if colored.shape[:2] != image.shape[:2]:
        colored = tf.image.resize(colored, image.shape[:2], method='bilinear').numpy().astype(np.uint8)
    overlay = colored[:, :, :3] * alpha + image * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
