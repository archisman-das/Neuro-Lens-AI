import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from src.models import get_model


def load_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.asarray(image, dtype=np.float32)
    return image_array


def parse_args():
    parser = argparse.ArgumentParser(description='Run prediction on a single MRI image')
    parser.add_argument('--model', choices=['cnn', 'transfer', 'vit'], default='cnn')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = get_model(args.model, transfer_weights=None)
    model.load_weights(args.weights)

    image = load_image(args.image)
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
    class_label = 'tumor' if prediction >= 0.5 else 'no_tumor'
    print(f'Image: {args.image}')
    print(f'Probability tumor: {prediction:.4f}')
    print(f'Predicted class: {class_label}')


if __name__ == '__main__':
    main()
