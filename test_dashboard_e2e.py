"""End-to-end smoke test for dashboard.py.

Starts the local HTTP server in a child process, then hits /metrics, /predict
(one call per classifier), and /segment with a real tumor MRI from
dataset_real/test/tumor/. Saves the returned base64 PNGs to test_outputs/ for
visual inspection and prints a per-endpoint pass/fail summary. Exits 0 on
full success, 1 on any failure. Always tears down the server.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import urllib.error

REPO_ROOT = Path(__file__).resolve().parent
PORT = 8511  # avoid colliding with a manually-run dashboard on 8501
BASE = f'http://127.0.0.1:{PORT}'
SAMPLE_IMAGE = REPO_ROOT / 'dataset_real' / 'test' / 'tumor'
OUTPUT_DIR = REPO_ROOT / 'test_outputs'


def find_sample_image() -> Path:
    """Return either the CLI-supplied --image or a fallback tumor slice."""
    for i, arg in enumerate(sys.argv):
        if arg in ('--image', '-i') and i + 1 < len(sys.argv):
            p = Path(sys.argv[i + 1])
            if not p.exists():
                raise FileNotFoundError(f'Provided --image does not exist: {p}')
            return p
    candidates = sorted([*SAMPLE_IMAGE.glob('*.jpg'), *SAMPLE_IMAGE.glob('*.png')])
    if not candidates:
        raise FileNotFoundError(f'No test MRI image found under {SAMPLE_IMAGE}')
    return candidates[0]


def wait_for_server(timeout_sec: float = 60.0) -> bool:
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'{BASE}/metrics', timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(0.5)
    print(f'  server never came up: {last_err}', flush=True)
    return False


def multipart_body(fields: dict, file_field: str, file_path: Path) -> tuple[bytes, str]:
    """Hand-rolled multipart/form-data so we don't add an HTTP-client dep."""
    boundary = '----neurolens-e2e-boundary-9f3a2c'
    lines = []
    for k, v in fields.items():
        lines.append(f'--{boundary}'.encode())
        lines.append(f'Content-Disposition: form-data; name="{k}"'.encode())
        lines.append(b'')
        lines.append(str(v).encode())
    lines.append(f'--{boundary}'.encode())
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"'.encode()
    )
    lines.append(b'Content-Type: image/jpeg')
    lines.append(b'')
    lines.append(file_path.read_bytes())
    lines.append(f'--{boundary}--'.encode())
    lines.append(b'')
    body = b'\r\n'.join(lines)
    return body, f'multipart/form-data; boundary={boundary}'


def http_post_multipart(url: str, fields: dict, file_field: str, file_path: Path, timeout=300):
    body, ctype = multipart_body(fields, file_field, file_path)
    req = urllib.request.Request(url, data=body, method='POST', headers={'Content-Type': ctype})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def http_get_json(url: str, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))


def decode_data_url_to(path: Path, data_url: str) -> int:
    """Decode a data:image/png;base64,... data URL to a PNG file. Returns bytes written."""
    assert data_url.startswith('data:image/'), f'unexpected data URL: {data_url[:60]}'
    payload = data_url.split(',', 1)[1]
    blob = base64.b64decode(payload)
    path.write_bytes(blob)
    return len(blob)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = find_sample_image()
    print(f'[setup] using sample image: {image_path}', flush=True)

    print(f'[start] launching dashboard on port {PORT}...', flush=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / 'dashboard.py'), '--port', str(PORT)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    failures = []
    try:
        if not wait_for_server():
            return 1
        print('[start] server is up.\n', flush=True)

        # 1. /metrics
        print('[test 1/3] GET /metrics', flush=True)
        try:
            status, payload = http_get_json(f'{BASE}/metrics')
            assert status == 200, f'status {status}'
            for m in ['cnn', 'transfer', 'vit']:
                assert m in payload, f'missing model {m}'
                e = payload[m]
                assert e['weights_found'], f'weights not found for {m}'
                assert e['metrics_found'], f'metrics not found for {m}'
                acc = e['metrics']['accuracy']
                auc = e['metrics']['roc_auc']
                print(f'  {m}: weights+metrics OK   accuracy={acc:.4f}  roc_auc={auc:.4f}')
            (OUTPUT_DIR / 'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
            print('  -> saved test_outputs/metrics.json\n', flush=True)
        except Exception as exc:
            failures.append(('GET /metrics', exc))
            print(f'  FAIL: {exc}\n', flush=True)

        # 2. /segment (run first because it's GPU-backed and fast; the slow
        # ViT predict in step 3 would otherwise block this on the single-
        # threaded HTTP server).
        print('[test 2/3] POST /segment', flush=True)
        try:
            status, blob = http_post_multipart(
                f'{BASE}/segment',
                {'threshold': '0.5'},
                'image',
                image_path,
            )
            assert status == 200, f'status {status}'
            payload = json.loads(blob.decode('utf-8'))
            assert payload.get('success') is True, f'success=False: {payload}'
            print(f'  model={payload["model"]}  source_dir={payload.get("source_dir")}'
                  f'  threshold={payload["threshold"]}'
                  f'  image_size={payload.get("image_size")}'
                  f'  tumor_area_px={payload["tumor_area_px"]}')
            n_mask = decode_data_url_to(OUTPUT_DIR / 'segmentation_mask.png', payload['mask'])
            n_over = decode_data_url_to(OUTPUT_DIR / 'segmentation_overlay.png', payload['overlay'])
            print(f'  -> saved mask ({n_mask} bytes) and overlay ({n_over} bytes)\n', flush=True)
        except Exception as exc:
            failures.append(('POST /segment', exc))
            print(f'  FAIL: {exc}\n', flush=True)

        # 3. /predict for each classifier
        print('[test 3/3] POST /predict (per model)', flush=True)
        for model_name in ['cnn', 'transfer', 'vit']:
            try:
                status, blob = http_post_multipart(
                    f'{BASE}/predict',
                    {'model': model_name},
                    'image',
                    image_path,
                )
                assert status == 200, f'status {status}'
                payload = json.loads(blob.decode('utf-8'))
                assert payload.get('success') is True, f'success=False: {payload}'
                r = payload['result']
                prob = r['probability']
                label = r['label']
                print(f'  {model_name}: prob={prob:.4f}  label={label}  weights={r.get("weights")}')
                # Save Grad-CAM and uploaded image for the convolutional models.
                if r.get('gradcam'):
                    n = decode_data_url_to(OUTPUT_DIR / f'gradcam_{model_name}.png', r['gradcam'])
                    print(f'    -> saved test_outputs/gradcam_{model_name}.png ({n} bytes)')
                else:
                    print('    no Grad-CAM (expected for ViT)')
                if r.get('image'):
                    decode_data_url_to(OUTPUT_DIR / f'input_{model_name}.png', r['image'])
            except Exception as exc:
                failures.append((f'POST /predict ({model_name})', exc))
                print(f'  FAIL: {exc}')
        print('', flush=True)

    finally:
        print('[teardown] stopping dashboard...', flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print('\n=== SUMMARY ===', flush=True)
    if failures:
        print(f'FAIL ({len(failures)} failure{"s" if len(failures) != 1 else ""}):')
        for name, exc in failures:
            print(f'  - {name}: {exc}')
        return 1
    print('PASS: all endpoints returned valid responses.')
    print(f'Outputs in {OUTPUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
