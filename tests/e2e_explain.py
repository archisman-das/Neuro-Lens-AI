"""E2E test of the /explain pipeline.

Hits build_explanation() directly (skipping HTTP) on a known tumor sample.
Two passes:
  1. backend='none'   -> validates segmentation + classifier + feature extractor
                         (deterministic fallback narrative; no LLM call).
  2. backend='ollama' -> validates real LLM call against qwen2.5vl:7b.

Saves the response JSON to e2e_explain_<backend>.json for inspection and prints
a one-line PASS/FAIL summary per stage.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import build_explanation  # noqa: E402


def _sample_bytes() -> bytes:
    candidates = [
        ROOT / 'dataset_real' / 'test' / 'tumor' / 'tumor_00000.jpg',
        ROOT / 'dataset_real' / 'val' / 'tumor' / 'tumor_00000.jpg',
        ROOT / 'dataset_real' / 'train' / 'tumor' / 'tumor_00000.jpg',
    ]
    for p in candidates:
        if p.exists():
            print(f'[e2e] using sample: {p}')
            return p.read_bytes()
    raise FileNotFoundError('No tumor_00000.jpg found in dataset_real/{test,val,train}/tumor')


def _check(label: str, ok: bool, extra: str = ''):
    print(f'  [{"PASS" if ok else "FAIL"}] {label}{(" - " + extra) if extra else ""}')
    return ok


def run_one(image_bytes: bytes, backend: str | None):
    tag = backend or 'auto'
    print(f'\n=== /explain backend={tag} ===')
    t0 = time.time()
    result = build_explanation(image_bytes, threshold=0.5, modality=None, backend=backend)
    elapsed = time.time() - t0
    print(f'  elapsed: {elapsed:.1f}s')

    # Persist for inspection.
    out_path = ROOT / f'e2e_explain_{tag}.json'
    # Strip giant data URLs before writing to disk (kept in memory for assertions).
    redacted = json.loads(json.dumps(result, default=str))
    for k in ('mask', 'overlay'):
        if isinstance(redacted.get('segmentation'), dict) and k in redacted['segmentation']:
            redacted['segmentation'][k] = f'<data:image/png;base64 {len(str(result["segmentation"].get(k,"")))} chars>'
    if isinstance(redacted.get('classifiers'), dict):
        for name, c in redacted['classifiers'].items():
            if isinstance(c, dict):
                if c.get('gradcam'):
                    c['gradcam'] = f'<data:image/png;base64 {len(c["gradcam"])} chars>'
    out_path.write_text(json.dumps(redacted, indent=2, default=str), encoding='utf-8')
    print(f'  saved: {out_path.name}')

    ok = True
    ok &= _check('success', bool(result.get('success')), str(result.get('error')))
    seg = result.get('segmentation', {}) or {}
    ok &= _check('segmentation.success', bool(seg.get('success')))
    ok &= _check('segmentation has mask', bool(seg.get('mask')))
    ok &= _check('segmentation has overlay', bool(seg.get('overlay')))
    cls = result.get('classifiers', {}) or {}
    for m in ('cnn', 'transfer', 'vit'):
        c = cls.get(m, {}) or {}
        ok &= _check(f'classifier[{m}] probability', isinstance(c.get('probability'), (int, float)))
    feats = result.get('features', {}) or {}
    ok &= _check('features.geometry', 'geometry' in feats)
    ok &= _check('features.intensity_per_channel', 'intensity_per_channel' in feats)
    ok &= _check('features.texture', 'texture' in feats)
    # New medical features:
    ok &= _check('features.morphology', 'morphology' in feats)
    ok &= _check('features.mass_effect', 'mass_effect' in feats)
    ok &= _check('features.internal_architecture', 'internal_architecture' in feats)
    ok &= _check('features.grade_evidence', 'grade_evidence' in feats)
    ok &= _check('features.overall_confidence', 'overall_confidence' in feats)
    exp = result.get('explanation', {}) or {}
    ok &= _check(f'explanation.backend == "{tag}"', exp.get('backend') == (backend or exp.get('backend')),
                 f'got "{exp.get("backend")}"')
    ok &= _check('explanation.summary non-empty', bool(exp.get('summary')))
    ok &= _check('explanation.impression non-empty', bool(exp.get('impression')))
    ok &= _check('explanation.differential_with_citations',
                  isinstance(exp.get('differential_with_citations'), list))
    ok &= _check('explanation.recommendation non-empty', bool(exp.get('recommendation')))
    ok &= _check('explanation.hallucination_safety set', bool(exp.get('hallucination_safety')))
    if backend != 'none':
        ok &= _check('explanation.llm_passes set', isinstance(exp.get('llm_passes'), dict))
    return ok


def main():
    image_bytes = _sample_bytes()
    pass_none = run_one(image_bytes, 'none')
    pass_ollama = run_one(image_bytes, 'ollama')
    print('\n=== overall ===')
    print(f'  backend=none   : {"PASS" if pass_none else "FAIL"}')
    print(f'  backend=ollama : {"PASS" if pass_ollama else "FAIL"}')
    sys.exit(0 if (pass_none and pass_ollama) else 1)


if __name__ == '__main__':
    main()
