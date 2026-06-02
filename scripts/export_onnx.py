"""Export PyTorch checkpoints to ONNX for fast real-time inference.

Why: onnxruntime-gpu beats PyTorch on cold-start + single-image latency for
inference workloads (no autograd, no Python-level layer dispatch). On RTX 4060
the SMP UNet inference drops from ~85 ms to ~25 ms per 256x256 image. CPU is
the bigger win (3-4x faster than PyTorch CPU).

Coverage:
  - 3 classifiers (cnn / transfer / vit) from real_eval_current/<name>/best_weights.pt
  - 5 segmentation checkpoints in segmentation_artifacts/<dir>/best_model.pt

Outputs land next to the source checkpoint as best_model.onnx /
best_weights.onnx. A numerical-equivalence check runs against the original
PyTorch model with random input; the export fails loudly if max abs diff
exceeds tol (default 1e-3, since FP16 in some onnxruntime ops can drift).

Usage:
  python scripts/export_onnx.py                # exports all known checkpoints
  python scripts/export_onnx.py --skip-classifiers
  python scripts/export_onnx.py --models vit attention_unet_v3
  python scripts/export_onnx.py --tol 5e-3 --image-size 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


CLASSIFIER_NAMES = ['cnn', 'transfer', 'vit']
SEGMENTATION_DIRS = [
    'attention_unet_v5',
    'attention_unet_v3',
    'attention_unet_t1c',
    'attention_unet_v2',
    'attention_unet_lgg',
    'attention_unet',
]

# Segmentation checkpoints that store a bare state_dict (no architecture
# metadata). Map dir_name -> SMP encoder so we can rebuild the wrapper.
BARE_STATE_DICT_DIRS = {
    'attention_unet_v5': 'resnet34',  # train_segmentation_v5.py: SMP UNet + ResNet34
}


def _classifier_paths(name: str) -> tuple[Path, Path] | None:
    """Return (pt, onnx) destinations for a classifier, or None if not found."""
    for candidate_dir in ('real_eval_current', 'real_eval_fixed', 'artifacts'):
        pt = ROOT / candidate_dir / name / 'best_weights.pt'
        if pt.exists():
            return pt, pt.with_suffix('.onnx')
    return None


def _segmentation_paths(dir_name: str) -> tuple[Path, Path] | None:
    pt = ROOT / 'segmentation_artifacts' / dir_name / 'best_model.pt'
    if pt.exists():
        return pt, pt.with_suffix('.onnx')
    return None


def _load_classifier(name: str, ckpt_path: Path, device: torch.device):
    from src.classifier_torch import get_classifier
    model = get_classifier(name).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = state.get('state_dict', state)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _load_segmentation(ckpt_path: Path, device: torch.device):
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    # Bare state-dict checkpoints (v5+): no wrapper dict, just a tensor map.
    # The dir name disambiguates which SMP encoder was used.
    if isinstance(state, dict) and 'state_dict' not in state and 'architecture' not in state:
        dir_name = ckpt_path.parent.name
        encoder = BARE_STATE_DICT_DIRS.get(dir_name)
        if encoder is None:
            raise RuntimeError(
                f"bare state_dict checkpoint at {ckpt_path} has no architecture "
                f"metadata and {dir_name} is not in BARE_STATE_DICT_DIRS"
            )
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name=encoder, encoder_weights=None,
                          in_channels=3, classes=1).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model
    arch = state.get('architecture')
    encoder = state.get('encoder')
    if arch and encoder:
        import segmentation_models_pytorch as smp
        SmpClass = getattr(smp, arch)
        model = SmpClass(encoder_name=encoder, encoder_weights=None,
                          in_channels=3, classes=1).to(device)
    else:
        from src.segmentation_torch import AttentionUNet
        cfg = state.get('config', {}) or {}
        model = AttentionUNet(in_channels=3,
                               base_filters=int(cfg.get('base_filters', 32)),
                               dropout=float(cfg.get('dropout', 0.2))).to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model


def _export_one(model: torch.nn.Module, onnx_path: Path, image_size: int,
                  device: torch.device, dynamic_batch: bool = True) -> dict:
    """Run torch.onnx.export and report timings."""
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    t0 = time.time()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        {'input': {0: 'batch'}, 'output': {0: 'batch'}}
        if dynamic_batch else None
    )
    # dynamo=False forces the legacy tracing-based exporter. The new dynamo
    # exporter prints unicode progress markers to stdout that crash on Windows
    # charmap consoles, and silently rewrites the requested opset. The legacy
    # path is also the more production-tested route for SMP UNet + ResNet50.
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    elapsed = time.time() - t0
    return {
        'onnx_path': str(onnx_path),
        'export_seconds': elapsed,
        'size_mb': onnx_path.stat().st_size / 1e6,
    }


def _verify_numerical_equivalence(model: torch.nn.Module, onnx_path: Path,
                                    image_size: int, device: torch.device,
                                    tol: float) -> dict:
    """Run a forward pass through both PyTorch and ONNX with the same random
    input; report the max absolute difference."""
    import onnxruntime as ort
    np.random.seed(0)
    arr = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(arr).to(device))
        if isinstance(torch_out, (tuple, list)):
            torch_out = torch_out[0]
        torch_np = torch_out.detach().cpu().numpy()

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_out = sess.run(None, {'input': arr})[0]

    diff = np.abs(torch_np - onnx_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    if max_diff > tol:
        raise RuntimeError(
            f'ONNX export {onnx_path.name} failed numerical-equivalence check: '
            f'max abs diff {max_diff:.6f} > tol {tol}. mean {mean_diff:.6f}. '
            f'Check op support / opset version.'
        )
    return {'max_abs_diff': max_diff, 'mean_abs_diff': mean_diff,
            'check_provider': sess.get_providers()[0]}


def _bench_speed(onnx_path: Path, image_size: int, n_warmup: int = 3,
                  n_iters: int = 20) -> dict:
    """Quick latency benchmark on the exported ONNX (single-image)."""
    import onnxruntime as ort
    arr = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    out = {}
    for ep, label in [(['CUDAExecutionProvider', 'CPUExecutionProvider'], 'cuda'),
                       (['CPUExecutionProvider'], 'cpu')]:
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=ep)
            if label not in sess.get_providers()[0].lower():
                continue
            for _ in range(n_warmup):
                sess.run(None, {'input': arr})
            t0 = time.time()
            for _ in range(n_iters):
                sess.run(None, {'input': arr})
            ms = (time.time() - t0) / n_iters * 1000
            out[f'{label}_ms_per_image'] = round(ms, 2)
        except Exception as exc:
            out[f'{label}_error'] = f'{type(exc).__name__}: {exc}'
    return out


def export_classifier(name: str, image_size: int, tol: float,
                       device: torch.device) -> dict:
    paths = _classifier_paths(name)
    if paths is None:
        return {'name': name, 'status': 'skipped_no_pt', 'kind': 'classifier'}
    pt_path, onnx_path = paths
    print(f'[classifier:{name}] {pt_path}')
    model = _load_classifier(name, pt_path, device)
    info = _export_one(model, onnx_path, image_size, device)
    info.update(_verify_numerical_equivalence(model, onnx_path, image_size, device, tol))
    info.update(_bench_speed(onnx_path, image_size))
    return {'name': name, 'kind': 'classifier', 'status': 'ok', **info}


def export_segmentation(dir_name: str, image_size: int, tol: float,
                         device: torch.device) -> dict:
    paths = _segmentation_paths(dir_name)
    if paths is None:
        return {'name': dir_name, 'status': 'skipped_no_pt', 'kind': 'segmentation'}
    pt_path, onnx_path = paths
    print(f'[seg:{dir_name}] {pt_path}')
    model = _load_segmentation(pt_path, device)
    info = _export_one(model, onnx_path, image_size, device)
    info.update(_verify_numerical_equivalence(model, onnx_path, image_size, device, tol))
    info.update(_bench_speed(onnx_path, image_size))
    return {'name': dir_name, 'kind': 'segmentation', 'status': 'ok', **info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=None,
                     help='Whitelist of model names to export (classifier or seg dir). Default: all.')
    ap.add_argument('--skip-classifiers', action='store_true')
    ap.add_argument('--skip-segmentation', action='store_true')
    ap.add_argument('--image-size', type=int, default=None,
                     help='Override input H=W. Default: 224 for classifiers, 256 for seg.')
    ap.add_argument('--tol', type=float, default=1e-3,
                     help='Max abs diff between PyTorch and ONNX outputs to accept the export.')
    ap.add_argument('--device', default=None, help='torch device override (e.g. cuda, cpu).')
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[info] device={device}, onnxruntime providers will include CUDA + CPU.')

    results: list[dict] = []
    whitelist = set(args.models) if args.models else None

    if not args.skip_classifiers:
        for name in CLASSIFIER_NAMES:
            if whitelist and name not in whitelist:
                continue
            img = args.image_size or 224
            try:
                results.append(export_classifier(name, img, args.tol, device))
            except Exception as exc:
                results.append({'name': name, 'kind': 'classifier',
                                 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})

    if not args.skip_segmentation:
        for d in SEGMENTATION_DIRS:
            if whitelist and d not in whitelist:
                continue
            img = args.image_size or 256
            try:
                results.append(export_segmentation(d, img, args.tol, device))
            except Exception as exc:
                results.append({'name': d, 'kind': 'segmentation',
                                 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})

    # Pretty print summary.
    print('\n=== Export Summary ===')
    for r in results:
        name = r.get('name')
        kind = r.get('kind')
        status = r.get('status')
        if status == 'ok':
            print(f'  [OK]  {kind:<13} {name:<22} '
                   f'-> {Path(r["onnx_path"]).name}  '
                   f'{r["size_mb"]:.1f} MB  '
                   f'maxdiff={r["max_abs_diff"]:.2e}  '
                   f'cuda={r.get("cuda_ms_per_image", "n/a")}ms  '
                   f'cpu={r.get("cpu_ms_per_image", "n/a")}ms')
        elif status == 'skipped_no_pt':
            print(f'  [SKP] {kind:<13} {name:<22} (no .pt found)')
        else:
            print(f'  [ERR] {kind:<13} {name:<22} {r.get("error", "?")}')

    fails = [r for r in results if r.get('status') == 'error']
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
