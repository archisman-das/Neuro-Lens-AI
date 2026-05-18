import json
import os
import re
import sys
import urllib.parse
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
from pathlib import Path

from PIL import Image
import numpy as np
import base64
import io
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / 'web_dashboard'
MODEL_TYPES = ['cnn', 'transfer', 'vit']
MODEL_LABELS = {'cnn': 'CNN', 'transfer': 'Transfer Learning', 'vit': 'Vision Transformer'}
ARTIFACTS_DIRS = [ROOT_DIR / 'real_eval_fixed', ROOT_DIR / 'real_eval_current', ROOT_DIR / 'artifacts']
MODEL_CACHE = {}

sys.path.append(str(ROOT_DIR))


def find_weights_path(model_name):
    for artifacts_dir in ARTIFACTS_DIRS:
        model_dir = artifacts_dir / model_name
        if not model_dir.exists():
            continue
        candidates = [model_dir / 'best_weights.weights.h5', model_dir / 'best_weights.h5']
        for candidate in candidates:
            if candidate.exists():
                return candidate
        for candidate in model_dir.glob('*.weights.h5'):
            return candidate
    return None


def summarize_metrics(metrics):
    if not isinstance(metrics, dict):
        return None
    report = metrics.get('classification_report', {})
    accuracy = metrics.get('accuracy')
    if isinstance(report, dict):
        accuracy = accuracy or report.get('accuracy')
        weighted = report.get('weighted avg', report.get('weighted_avg', {}))
        matrix = metrics.get('confusion_matrix')
        confusion = None
        if isinstance(matrix, list) and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix):
            confusion = {
                'tn': int(matrix[0][0]),
                'fp': int(matrix[0][1]),
                'fn': int(matrix[1][0]),
                'tp': int(matrix[1][1]),
            }
        return {
            'accuracy': float(accuracy) if accuracy is not None else None,
            'precision': float(weighted.get('precision')) if weighted.get('precision') is not None else None,
            'recall': float(weighted.get('recall')) if weighted.get('recall') is not None else None,
            'f1_score': float(weighted.get('f1-score', weighted.get('f1_score'))) if weighted.get('f1-score', weighted.get('f1_score')) is not None else None,
            'roc_auc': float(metrics.get('roc_auc')) if metrics.get('roc_auc') is not None else None,
            'confusion_matrix': confusion,
        }
    return None


def load_model_metrics():
    data = {}
    for model_name in MODEL_TYPES:
        metrics_path = next(
            (artifacts_dir / f'{model_name}_evaluation_metrics.json'
             for artifacts_dir in ARTIFACTS_DIRS
             if (artifacts_dir / f'{model_name}_evaluation_metrics.json').exists()),
            None,
        )
        model_entry = {
            'model': model_name,
            'label': MODEL_LABELS[model_name],
            'weights_found': bool(find_weights_path(model_name)),
            'metrics_found': False,
            'metrics': None,
        }
        if metrics_path and metrics_path.exists():
            try:
                with metrics_path.open('r', encoding='utf-8') as fh:
                    metrics = json.load(fh)
                model_entry['metrics'] = summarize_metrics(metrics)
                model_entry['metrics_found'] = model_entry['metrics'] is not None
            except Exception:
                model_entry['metrics_found'] = False
        data[model_name] = model_entry
    return data


def predict_image(model_name, image_bytes):
    if model_name not in MODEL_TYPES and model_name != 'all':
        raise ValueError('Unknown model selected.')

    if model_name == 'all':
        results = {}
        for name in MODEL_TYPES:
            results[name] = predict_image(name, image_bytes)
        return results

    weights_path = find_weights_path(model_name)
    if not weights_path:
        return {
            'error': 'No trained weights found for this model.',
            'hint': f'Train {MODEL_LABELS[model_name]} and save weights in artifacts/{model_name}/best_weights.weights.h5.',
        }

    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    cache_key = (model_name, str(weights_path), weights_path.stat().st_mtime)
    model = MODEL_CACHE.get(cache_key)
    if model is None:
        MODEL_CACHE.clear()
        from src.models import get_model

        model = get_model(model_name, transfer_weights=None)
        model.load_weights(str(weights_path))
        MODEL_CACHE[cache_key] = model
    score = float(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0][0])
    label = 'tumor' if score >= 0.5 else 'no_tumor'
    # Prepare response payload
    result = {
        'probability': round(score, 4),
        'confidence': round(score if label == 'tumor' else 1.0 - score, 4),
        'label': label,
        'display_label': 'Tumor detected' if label == 'tumor' else 'No tumor detected',
        'weights': str(weights_path.name),
    }

    # Attach original uploaded image as data URL
    try:
        buf = io.BytesIO()
        Image.fromarray(image_array.astype('uint8')).save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        result['image'] = f'data:image/png;base64,{img_b64}'
    except Exception:
        result['image'] = None

    # Try to generate Grad-CAM for convolutional models
    try:
        from src.utils import make_gradcam_heatmap, overlay_heatmap
        conv_layer = None
        if model_name == 'cnn':
            conv_layer = 'conv_block_3'
        elif model_name == 'transfer':
            conv_layer = 'conv5_block3_out'

        if conv_layer is not None:
            heatmap = make_gradcam_heatmap(tf.expand_dims(image_array, axis=0), model, conv_layer)
            overlay = overlay_heatmap(image_array.astype('uint8'), heatmap)
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format='PNG')
            overlay_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            result['gradcam'] = f'data:image/png;base64,{overlay_b64}'
        else:
            result['gradcam'] = None
    except Exception:
        result['gradcam'] = None

    return result


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/metrics':
            self.respond_json(load_model_metrics())
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/predict':
            self.handle_predict()
            return
        self.send_error(404, 'Endpoint not found')

    def handle_predict(self):
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_error(400, 'Expected multipart/form-data')
            return

        boundary_match = re.search(r'boundary=(.+)', content_type)
        if not boundary_match:
            self.send_error(400, 'Missing boundary in Content-Type header')
            return

        boundary = boundary_match.group(1)
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
        boundary_bytes = boundary.encode('utf-8')

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        form = self.parse_multipart(body, boundary_bytes)

        model_name = form.get('model')
        file_item = form.get('image')
        if not model_name or not file_item or 'content' not in file_item:
            self.send_error(400, 'Missing model or image upload')
            return

        image_bytes = file_item['content']
        try:
            result = predict_image(model_name, image_bytes)
            self.respond_json({'success': True, 'result': result})
        except Exception as exc:
            self.respond_json({'success': False, 'error': str(exc)}, status=500)

    def parse_multipart(self, body, boundary):
        parts = body.split(b'--' + boundary)
        data = {}
        for part in parts:
            if not part or part in (b'--', b'--\r\n'):
                continue
            part = part.strip(b'\r\n')
            if not part:
                continue

            header_bytes, _, content = part.partition(b'\r\n\r\n')
            headers = {}
            for line in header_bytes.split(b'\r\n'):
                name, _, value = line.decode('utf-8', 'ignore').partition(':')
                headers[name.lower().strip()] = value.strip()

            disposition = headers.get('content-disposition', '')
            disposition_data = self.parse_content_disposition(disposition)
            name = disposition_data.get('name')
            if not name:
                continue

            if 'filename' in disposition_data:
                data[name] = {
                    'filename': disposition_data.get('filename'),
                    'content': content.rstrip(b'\r\n'),
                }
            else:
                data[name] = content.decode('utf-8', errors='replace').strip()
        return data

    def parse_content_disposition(self, disposition):
        values = {}
        parts = [part.strip() for part in disposition.split(';') if part.strip()]
        for part in parts:
            if '=' in part:
                key, val = part.split('=', 1)
                values[key.strip().lower()] = val.strip('"')
        return values

    def respond_json(self, data, status=200):
        payload = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def run(port=8501):
    if not WEB_DIR.exists():
        raise FileNotFoundError(f'Web dashboard files not found: {WEB_DIR}')

    address = ('', port)
    server = HTTPServer(address, DashboardHandler)
    url = f'http://localhost:{port}/'
    print(f'NeuroLens AI dashboard running at {url}')
    print('Open this URL in your browser. Press Ctrl+C here to stop the server.')
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the NeuroLens AI HTML dashboard')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    run(args.port)
