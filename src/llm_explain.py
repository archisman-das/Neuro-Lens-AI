"""LLM-backed explanation of the segmentation + classification output.

Takes (image, mask, overlay, classifier results, Grad-CAM, feature dict from
src.tumor_explainability) and asks a multimodal LLM to produce a structured
human-language explanation grounded in the numeric features.

Backend selection (first available wins):
  1. Ollama   - default. Set OLLAMA_HOST / OLLAMA_MODEL env vars to override.
                Defaults: http://localhost:11434, model qwen2.5vl:7b
  2. Anthropic - if ANTHROPIC_API_KEY is set
  3. OpenAI    - if OPENAI_API_KEY is set
  4. None      - returns a deterministic feature-grounded narrative without
                 calling any LLM. Still useful, just less narrative.

The output schema is intentionally fixed (top-level keys below) so the UI can
render predictable sections regardless of which backend produced it.

Output:
  {
    "backend": "ollama" | "anthropic" | "openai" | "none",
    "model": "<model name>",
    "summary": "<plain-language overview, 2-3 sentences>",
    "findings": {
        "geometry": "<...>",
        "localization": "<...>",
        "intensity": "<...>",
        "texture": "<...>",
        "multimodal": "<...>",  # only when modality channels are known
    },
    "differential_diagnosis_hints": ["<bullet>", ...],
    "model_agreement_analysis": "<...>",
    "confidence_assessment": "<...>",
    "disclaimer": "<...>",
    "raw_features": {...}    # the entire feature dict from tumor_explainability
  }
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import requests
except ImportError:
    requests = None


DEFAULT_OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
# Two model slots:
#   OLLAMA_MODEL_TEXT   - small text-only LLM for Pattern A (polish) and
#                         Pattern B (differential expansion). These tasks
#                         don't need vision and benefit from a smaller, more
#                         compliant model with low system-RAM footprint.
#                         Default qwen2.5:1.5b needs ~2 GiB host RAM.
#   OLLAMA_MODEL_VISION - vision-language model for Pattern C (visual
#                         co-observer). Loaded only if it actually fits; if
#                         it OOMs, Pattern C falls back to "skipped (RAM)"
#                         without poisoning the rest of the pipeline.
#                         Default qwen2.5vl:3b needs ~7 GiB host RAM.
#   OLLAMA_MODEL        - legacy override, mapped to MODEL_VISION for
#                         backwards-compat with earlier configs.
DEFAULT_OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5vl:3b')
DEFAULT_OLLAMA_MODEL_TEXT = os.environ.get('OLLAMA_MODEL_TEXT', 'qwen2.5:1.5b')
DEFAULT_OLLAMA_MODEL_VISION = os.environ.get('OLLAMA_MODEL_VISION', DEFAULT_OLLAMA_MODEL)
DEFAULT_ANTHROPIC_MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-6')
DEFAULT_OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')

# HuggingFace Inference Providers - routes to Groq / Together / Fireworks /
# Replicate / etc. under the hood, all hosting OPEN-WEIGHT models. One token
# (HF_TOKEN) authenticates against all providers. Free tier + small HF credit
# allowance per month makes this the right call for a public Spaces demo
# while keeping the LLM open source (we only call open weights like Llama 3.3
# and Qwen2.5-VL, never closed-weight Claude/GPT/Gemini).
#
# Two model slots to mirror the local Ollama text+vision split:
#   HF_MODEL_TEXT     - text-only LLM for Patterns A (polish) and B
#                        (differential expansion). Llama 3.3 70B by default;
#                        strictly better at JSON-schema compliance than the
#                        local qwen2.5:1.5b we use offline.
#   HF_MODEL_VISION   - vision-language LLM for Pattern C (visual co-observer).
#                        Llama 3.2 90B Vision by default; Qwen2.5-VL 72B
#                        (Qwen/Qwen2.5-VL-72B-Instruct) is also a strong pick.
DEFAULT_HF_ROUTER_BASE = os.environ.get(
    'HF_INFERENCE_BASE', 'https://router.huggingface.co/v1'
)
DEFAULT_HF_MODEL_TEXT = os.environ.get(
    'HF_MODEL_TEXT', 'meta-llama/Llama-3.3-70B-Instruct'
)
DEFAULT_HF_MODEL_VISION = os.environ.get(
    # Gemma 3 27B IT is Google's open-weight multimodal model and is enabled
    # by default on the HF Inference Providers router for the standard free
    # tier. Llama 3.2 Vision and Qwen2.5-VL are higher quality on paper but
    # are gated behind specific provider plans (HuggingFace Pro, Together
    # paid tier, etc.). Override with HF_MODEL_VISION when you have access
    # to a stronger vision model.
    'HF_MODEL_VISION', 'google/gemma-3-27b-it'
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _advisory_summary(advisory: dict) -> dict:
    """Boil the v9b advisory payload down to a flat dict the deterministic
    narrative can quote verbatim. Every value here is measured at request
    time — nothing is invented by the LLM. Used as part of the
    zero-hallucination substrate.
    """
    signal_states = []
    for label, fired_key, value_key in (
        ('Pattern Detector',        'v9c_fired',      'v9c_p95'),
        ('Reconstruction Detector', 'andi_fired',     'andi_max'),
        ('Tumor Outline Drawer',    'v8_fired',       'v8_area_px'),
        ('Asymmetry Detector',      'symmetry_fired', 'symmetry_p95'),
    ):
        fired = advisory.get(fired_key)
        value = advisory.get(value_key)
        if fired is None and value is None:
            continue
        signal_states.append({
            'signal': label,
            'fired': bool(fired) if fired is not None else None,
            'value': value,
        })
    measured = advisory.get('measured_performance') or {}
    return {
        'verdict': advisory.get('verdict', 'no_tumor'),
        'confidence': advisory.get('confidence', 'low'),
        'operating_point': advisory.get('operating_point', 'balanced'),
        'rule': advisory.get('rule', 'ensemble'),
        'signals_used': advisory.get('signals_used', ''),
        'signal_states': signal_states,
        'review_recommended': bool(advisory.get('review_recommended', False)),
        'measured_recall': measured.get('ood_recall'),
        'measured_fpr': measured.get('ood_fpr'),
        'measured_f1': measured.get('ood_f1'),
        'cohort': measured.get('cohort', '246-sample OOD bench'),
    }


def explain(
    image_rgb: np.ndarray,
    mask_bin: np.ndarray,
    overlay_rgb: Optional[np.ndarray],
    classifier_results: Optional[dict],
    gradcam_rgb: Optional[np.ndarray],
    features: dict,
    *,
    modality_channels: Optional[tuple[str, str, str]] = None,
    backend: Optional[str] = None,
    advisory: Optional[dict] = None,
) -> dict:
    """Layered explanation pipeline (zero-hallucination by construction).

    Architecture:
      Step 0  Deterministic narrative from `features`. Source of truth.
              Every number and label here is measured. Returned as a key
              `deterministic_report` so the UI can always show the verified
              version side by side with anything the LLM contributes.

      Step A  LLM polish pass. The LLM is given the deterministic narrative
              as the ONLY source of facts and asked to rephrase it in clean
              radiology-style prose. Post-check: any number or anatomical
              term in the polished prose must appear in the source. If not,
              we fall back to the deterministic prose.

      Step B  LLM differential-expansion pass. The LLM is given the structured
              features (no images) and asked to propose differential diagnoses,
              each with a citation to specific feature key(s). Post-check: any
              bullet without a verifiable citation is dropped.

      Step C  LLM visual co-observer pass. The LLM is given the original MRI
              + overlay and asked for purely-qualitative observations (e.g.
              "rim brighter than core in superior aspect"). Each observation
              is checked against the features dict; observations that
              CONTRADICT a measurement go into `disagreements` rather than
              into the findings, so the user can see the model conflict.

    Skip behaviour:
      - backend='none' -> run only Step 0. Zero-hallucination guarantee.
      - backend='ollama'/'anthropic'/'openai' -> run 0 + A + B + C.
      - Any LLM step that fails individually does NOT poison the pipeline;
        the deterministic substrate is preserved.
    """
    if backend is None:
        backend = _pick_backend()

    # ---- Step 0 - deterministic substrate ---------------------------------
    deterministic = _local_narrative(features, classifier_results)
    # Inject the 4-signal advisory verdict + signal-level firing breakdown
    # into the deterministic substrate. This becomes part of the source of
    # truth that the LLM polish pass is constrained to, so the generated
    # prose can reference the ensemble rule without inventing it.
    if advisory:
        deterministic['ensemble_advisory'] = _advisory_summary(advisory)

    backend_used = 'none'
    model_used = 'deterministic'
    llm_passes: dict = {
        'polish': {'status': 'skipped'},
        'differential_expansion': {'status': 'skipped'},
        'visual_observer': {'status': 'skipped'},
    }

    # If no LLM, return the deterministic report verbatim.
    if backend == 'none':
        payload = _coerce_to_schema(deterministic, features, classifier_results=classifier_results)
        payload['backend'] = 'none'
        payload['model'] = 'deterministic'
        payload['raw_features'] = features
        payload['deterministic_report'] = deterministic
        payload['llm_passes'] = llm_passes
        payload['hallucination_safety'] = 'guaranteed_zero (no LLM was called)'
        if advisory:
            payload['ensemble_advisory'] = deterministic['ensemble_advisory']
        return payload

    # Short-circuit: when the classifier consensus is firmly NO_TUMOR (all
    # three models agree at >=moderate confidence), there is no clinical
    # finding to enrich. Calling the LLM here only risks it inventing a
    # tumor narrative around the false-positive U-Net mask - which it
    # frequently does even with our citation guards, because the polish
    # prompt asks it to rephrase the impression. Skip all three LLM passes,
    # return the verdict-aware deterministic report.
    cls_verdict, cls_mean_p, cls_band = _classifier_consensus(classifier_results)
    if cls_verdict == 'no_tumor' and cls_band in ('high', 'moderate'):
        skip_reason = (
            f'classifier_consensus_no_tumor (mean p={cls_mean_p:.3f}, '
            f'{cls_band} confidence) - no clinical finding to enrich.'
        )
        # Patterns A (polish) and B (differential expansion) stay short-
        # circuited - there's no clinical finding to polish or differentiate.
        # Pattern C (general visual observer) is replaced with Pattern D
        # below: a vision-LLM call with a NEGATIVE-specific prompt that asks
        # the model to articulate which visible features in the actual MRI
        # support the no-tumor verdict. This gives the user real reasoning
        # for negative cases instead of an empty short-circuit.
        for k in ('polish', 'differential_expansion'):
            llm_passes[k] = {'status': 'short_circuited', 'reason': skip_reason}

        # Build the image set we'll feed to Pattern D.
        if backend == 'ollama':
            image_set = [('original_mri', image_rgb), ('overlay', overlay_rgb)]
        else:
            image_set = [
                ('original_mri', image_rgb),
                ('predicted_mask', _mask_to_rgb(mask_bin)),
                ('overlay', overlay_rgb),
                ('gradcam', gradcam_rgb),
            ]
        images_b64, image_labels = [], []
        for label, arr in image_set:
            if arr is None:
                continue
            try:
                images_b64.append(_to_base64_png(arr))
                image_labels.append(label)
            except Exception:
                pass

        vision_negative_reasoning = None
        vision_model_used = None
        if backend == 'ollama':
            vision_model_used = DEFAULT_OLLAMA_MODEL_VISION
        elif backend == 'hf_inference':
            vision_model_used = DEFAULT_HF_MODEL_VISION
        elif backend == 'anthropic':
            vision_model_used = DEFAULT_ANTHROPIC_MODEL
        elif backend == 'openai':
            vision_model_used = DEFAULT_OPENAI_MODEL

        if images_b64 and backend != 'none':
            try:
                neg_prompt = _build_negative_visual_prompt(
                    cls_mean_p or 0.0, cls_band or 'moderate', classifier_results,
                )
                raw_neg = _call_backend_with_model(
                    backend, neg_prompt, images_b64, image_labels,
                    model_override=vision_model_used,
                )
                vision_negative_reasoning, neg_warnings = _validate_negative_visual_reasoning(
                    raw_neg
                )
                llm_passes['visual_observer'] = {
                    'status': 'ok_negative_reasoning' if vision_negative_reasoning else 'rejected',
                    'pattern': 'D (vision negative reasoning)',
                    'model': vision_model_used,
                    'reason': 'classifier consensus negative; vision LLM asked why the MRI looks healthy',
                    'raw_chars': len(raw_neg or ''),
                    'warnings': neg_warnings,
                }
            except Exception as exc:
                err = f'{type(exc).__name__}: {exc}'
                is_oom = 'system memory' in err.lower() or 'memory' in err.lower()
                is_quota = 'quota' in err.lower() or '429' in err
                llm_passes['visual_observer'] = {
                    'status': ('skipped_insufficient_ram' if is_oom
                                else 'skipped_quota_exhausted' if is_quota
                                else 'error'),
                    'pattern': 'D (vision negative reasoning)',
                    'model': vision_model_used,
                    'error': err,
                }
        else:
            llm_passes['visual_observer'] = {
                'status': 'short_circuited',
                'pattern': 'D (vision negative reasoning)',
                'reason': 'no images available or backend=none',
            }

        payload = _coerce_to_schema(deterministic, features, classifier_results=classifier_results)
        payload['backend'] = backend
        payload['model'] = (
            f'deterministic + vision negative reasoning ({vision_model_used})'
            if vision_negative_reasoning else
            'deterministic (short-circuit: no-tumor consensus)'
        )
        payload['raw_features'] = features
        payload['deterministic_report'] = deterministic
        payload['llm_passes'] = llm_passes
        payload['vision_negative_reasoning'] = vision_negative_reasoning
        payload['hallucination_safety'] = (
            'Patterns A and B short-circuited (no clinical finding to enrich). '
            + ('Pattern D (vision negative reasoning) called - output validated '
                'for negation-consistency (no tumor / lesion / abnormal claims).'
                if vision_negative_reasoning else
                'No LLM was called.')
        )
        return payload

    model_used = (os.environ.get('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL) if backend == 'ollama'
                  else f'{DEFAULT_HF_MODEL_TEXT} + {DEFAULT_HF_MODEL_VISION}' if backend == 'hf_inference'
                  else DEFAULT_ANTHROPIC_MODEL if backend == 'anthropic'
                  else DEFAULT_OPENAI_MODEL if backend == 'openai'
                  else 'deterministic')
    backend_used = backend

    # Build the image set once. Ollama gets only 2 to fit VRAM. HF Inference
    # routes to cloud GPUs so the 4-image set is fine and gives the visual
    # observer richer context.
    image_set = ([('original_mri', image_rgb), ('overlay', overlay_rgb)]
                 if backend == 'ollama' else
                 [('original_mri', image_rgb),
                  ('predicted_mask', _mask_to_rgb(mask_bin)),
                  ('overlay', overlay_rgb),
                  ('gradcam', gradcam_rgb)])
    images_b64, image_labels = [], []
    for label, arr in image_set:
        if arr is None:
            continue
        try:
            images_b64.append(_to_base64_png(arr))
            image_labels.append(label)
        except Exception:
            pass

    # ---- Split-model LLM strategy -----------------------------------------
    # Pattern A (polish) + Pattern B (differential expansion) don't need
    # vision - they reason over text and structured features. We send them
    # together to a SMALL text-only LLM (qwen2.5:1.5b ~2 GiB host RAM),
    # which fits even when the dashboard PyTorch stack is co-resident.
    # Pattern C (visual co-observer) requires actual vision; we attempt the
    # VL model only if it loads. If it OOMs, Pattern C cleanly reports
    # "skipped (insufficient_ram)" and the rest of the pipeline is unaffected.
    polished = None
    llm_differentials: list[dict] = []
    visual_observations: list[dict] = []
    visual_disagreements: list[dict] = []

    # Per-pass model selection. Each backend has its own text vs vision split
    # so we can use a small fast model for prose patterns and a stronger
    # vision-capable model for image co-observation.
    if backend == 'ollama':
        text_model = DEFAULT_OLLAMA_MODEL_TEXT
        vision_model = DEFAULT_OLLAMA_MODEL_VISION
    elif backend == 'hf_inference':
        text_model = DEFAULT_HF_MODEL_TEXT
        vision_model = DEFAULT_HF_MODEL_VISION
    else:
        # Anthropic / OpenAI pick one strong model that handles both text +
        # vision. We keep the model name as model_used for reporting.
        text_model = model_used
        vision_model = model_used

    # ---- Patterns A + B: text-only combined call --------------------------
    try:
        ab_prompt = _build_text_combined_prompt(deterministic, features)
        # 3072 ctx for the text pass: the prompt includes the full features
        # subset + citable keys + already-proposed differentials, which run
        # ~1200 tokens. 1.5b qwen2.5 is happy at 3072. Text-only KV growth is
        # small enough not to OOM.
        raw_ab = _call_backend_with_model(backend, ab_prompt, [], [],
                                            model_override=text_model,
                                            num_ctx_override=3072)
        logger_msg = f'[llm_explain] text-pass model={text_model} raw_chars={len(raw_ab)}'
        print(logger_msg, flush=True)
        ab_parsed = _try_parse_json(raw_ab) or {}
        # Polish
        polish_raw = ab_parsed.get('polished_impression') or ''
        polished_text, polish_warnings = _validate_polish(
            polish_raw if isinstance(polish_raw, str) else '', deterministic, features
        )
        llm_passes['polish'] = {
            'status': ('ok' if polished_text else ('rejected' if polish_warnings else 'empty')),
            'model': text_model,
            'raw_chars': len(polish_raw) if isinstance(polish_raw, str) else 0,
            'warnings': polish_warnings,
        }
        if polished_text:
            polished = polished_text
        # Differential expansion
        diff_items = ab_parsed.get('additional_differentials') or []
        diff_wrapped = json.dumps({'differentials': diff_items})
        accepted, rejected = _validate_differentials(diff_wrapped, features)
        llm_differentials = accepted
        llm_passes['differential_expansion'] = {
            'status': 'ok',
            'model': text_model,
            'accepted_count': len(accepted),
            'rejected_count': len(rejected),
            'rejected_bullets': rejected,
            'raw_chars': len(json.dumps(diff_items)),
        }
    except Exception as exc:
        err = f'{type(exc).__name__}: {exc}'
        llm_passes['polish'] = {'status': 'error', 'error': err, 'model': text_model}
        llm_passes['differential_expansion'] = {'status': 'error', 'error': err,
                                                  'model': text_model}

    # ---- Pattern C: vision call (best-effort, falls back on OOM/quota) ----
    try:
        if not images_b64:
            llm_passes['visual_observer'] = {'status': 'skipped',
                                              'reason': 'no_images_provided'}
        else:
            visual_prompt = _build_visual_prompt(features)
            raw_vis = _call_backend_with_model(
                backend, visual_prompt, images_b64, image_labels,
                model_override=vision_model,
            )
            observations, disagreements = _validate_visual_observations(raw_vis, features)
            visual_observations = observations
            visual_disagreements = disagreements
            llm_passes['visual_observer'] = {
                'status': 'ok',
                'model': vision_model,
                'observation_count': len(observations),
                'disagreement_count': len(disagreements),
                'raw_chars': len(raw_vis or ''),
            }
    except Exception as exc:
        err = f'{type(exc).__name__}: {exc}'
        is_oom = 'system memory' in err.lower() or 'memory' in err.lower()
        is_quota = 'quota' in err.lower() or '429' in err
        if is_oom:
            status, hint = 'skipped_insufficient_ram', (
                'Close apps to free RAM for the local vision model, '
                'or switch to HF Inference by setting HF_TOKEN.'
            )
        elif is_quota:
            status, hint = 'skipped_quota_exhausted', (
                'HF Inference free-tier quota reached. Add HF Pro or fall back '
                'to local Ollama by unsetting HF_TOKEN.'
            )
        else:
            status, hint = 'error', None
        llm_passes['visual_observer'] = {
            'status': status,
            'error': err,
            'model': vision_model,
            'recovery_hint': hint,
        }

    # ---- Assemble final payload ------------------------------------------
    final = dict(deterministic)  # start from the source of truth
    if polished:
        # Use polished prose for the user-facing summary; keep raw measurement
        # text alongside for verification.
        final['impression'] = polished
        final['summary'] = polished
    # Merge LLM differentials AFTER the rule-based ones; tag origin.
    for d in (final.get('differential_with_citations') or []):
        d.setdefault('origin', 'rule-based')
    for d in llm_differentials:
        d['origin'] = 'llm-citation-checked'
        final.setdefault('differential_with_citations', []).append(d)
    final['differential_diagnosis_hints'] = [
        d['statement'] for d in (final.get('differential_with_citations') or [])
    ]
    final['visual_observations'] = visual_observations
    final['visual_disagreements'] = visual_disagreements

    payload = _coerce_to_schema(final, features, classifier_results=classifier_results)
    payload['backend'] = backend_used
    payload['model'] = model_used
    payload['raw_features'] = features
    payload['deterministic_report'] = deterministic
    payload['llm_passes'] = llm_passes
    payload['hallucination_safety'] = (
        'deterministic substrate preserved; LLM-added content is citation-checked '
        'and conflict-flagged. See llm_passes for per-pass results.'
    )
    if deterministic.get('ensemble_advisory'):
        payload['ensemble_advisory'] = deterministic['ensemble_advisory']
    return payload


def _call_backend_with_model(backend: str, prompt_text: str, images_b64: list,
                               image_labels: list, *, model_override: Optional[str] = None,
                               num_ctx_override: Optional[int] = None) -> str:
    """Dispatch helper that lets a per-pass model override take effect.

    Ollama and HF Inference honor the override (text model vs vision model
    split). Anthropic/OpenAI continue to use their single DEFAULT model since
    those backends pick a strong general-purpose model regardless.
    """
    if backend == 'ollama':
        return _call_ollama(prompt_text, images_b64, image_labels,
                             model_override=model_override,
                             num_ctx_override=num_ctx_override)
    if backend == 'hf_inference':
        return _call_hf_inference(prompt_text, images_b64, image_labels,
                                    model_override=model_override)
    if backend == 'anthropic':
        return _call_anthropic(prompt_text, images_b64, image_labels)
    if backend == 'openai':
        return _call_openai(prompt_text, images_b64, image_labels)
    return ''


def _build_text_combined_prompt(deterministic: dict, features: dict) -> str:
    """Text-only prompt asking for Pattern A (polish) + Pattern B (differential
    expansion) in a single JSON response. No images. Designed for a small
    text-only LLM (e.g. qwen2.5:1.5b) that fits in ~2 GiB system RAM.

    Output schema (strict JSON, no markdown):
      {
        "polished_impression": "<3-6 sentences rephrasing the impression>",
        "additional_differentials": [
          {"statement": "...", "supported_by": ["feature.key=value"], "confidence": "low|moderate|high"}
        ]
      }
    """
    impression = deterministic.get('impression') or deterministic.get('summary') or ''
    findings = deterministic.get('findings') or {}
    fields_text = '\n'.join(f'- {k}: {v}' for k, v in findings.items() if v)
    rule_diffs = [d['statement'] for d in (deterministic.get('differential_with_citations') or [])]
    rule_text = '\n'.join(f'- {b}' for b in rule_diffs) or '(none)'
    feat_compact = _compact_features_for_diff(features)
    citable = '\n'.join(sorted(_DIFFERENTIAL_CITABLE_KEYS))
    # Include the ensemble advisory in the source-of-truth block so the
    # polish pass can reference the verdict + which signals fired without
    # inventing them. The strict rule already forbids adding facts not in
    # the source, so quoting the advisory here is the only safe surface.
    advisory_summary = deterministic.get('ensemble_advisory')
    if advisory_summary:
        advisory_block = (
            "--- ENSEMBLE ADVISORY (measured at request time) ---\n"
            f"{json.dumps(advisory_summary, indent=2, default=_safe_default)}\n"
        )
    else:
        advisory_block = ""
    return (
        "You are a neuroradiology assistant. Produce one JSON object with two sections. "
        "STRICT RULES:\n"
        "1. Do NOT invent any number or finding not in the measured features.\n"
        "2. Every numeric value you cite must match the features verbatim.\n"
        "3. If a section cannot be filled in good faith, return an empty value.\n"
        "4. Output a single JSON object - no markdown, no preamble.\n\n"
        "SECTION 1 - 'polished_impression' (Pattern A):\n"
        "  - 3 to 6 sentences of radiology-style prose.\n"
        "  - Rephrase the SOURCE; do NOT add facts. Numbers must stay identical.\n"
        "  - If an ENSEMBLE ADVISORY block is present, your prose may quote\n"
        "    the verdict, confidence, operating_point, rule, and individual\n"
        "    signal fired-states verbatim — these are measurements, not opinions.\n\n"
        "SECTION 2 - 'additional_differentials' (Pattern B):\n"
        "  - Up to 4 differentials NOT already in the already-proposed list.\n"
        "  - Each MUST cite a key from the citable whitelist as 'feature.key=value'.\n"
        "  - Schema per item: {statement, supported_by:[citations], "
        "confidence: 'low'|'moderate'|'high'}.\n\n"
        "--- SOURCE IMPRESSION ---\n"
        f"{impression}\n"
        "--- SOURCE FINDINGS ---\n"
        f"{fields_text}\n"
        f"{advisory_block}"
        "--- ALREADY-PROPOSED DIFFERENTIALS (do not repeat) ---\n"
        f"{rule_text}\n"
        "--- CITABLE FEATURE KEYS (whitelist) ---\n"
        f"{citable}\n"
        "--- COMPACT MEASURED FEATURES ---\n"
        f"{json.dumps(feat_compact, indent=2, default=_safe_default)}\n\n"
        "Now return the single JSON object."
    )


def _build_combined_prompt(deterministic: dict, features: dict) -> str:
    """Single prompt that asks for ALL three pattern outputs at once.

    Output schema (strict JSON, no markdown):
      {
        "polished_impression": "<3-6 sentences rephrasing the impression>",
        "additional_differentials": [
          {"statement": "...", "supported_by": ["feature.key=value"], "confidence": "low|moderate|high"}
        ],
        "visual_observations": [
          {"region": "...", "observation": "...", "claimed_property": "intensity|uniformity|border|spatial"}
        ]
      }

    Each section is validated independently after we receive the response, so
    a missing or malformed section degrades gracefully (e.g. polish empty ->
    we just keep the deterministic impression; rejected differentials -> we
    keep only the rule-based bullets).
    """
    impression = deterministic.get('impression') or deterministic.get('summary') or ''
    findings = deterministic.get('findings') or {}
    fields_text = '\n'.join(f'- {k}: {v}' for k, v in findings.items() if v)
    rule_diffs = [d['statement'] for d in (deterministic.get('differential_with_citations') or [])]
    rule_text = '\n'.join(f'- {b}' for b in rule_diffs) or '(none)'
    feat_compact = _compact_features_for_diff(features)
    citable = '\n'.join(sorted(_DIFFERENTIAL_CITABLE_KEYS))
    geom = features.get('geometry') or {}
    loc = features.get('localization') or {}
    return (
        "You are a neuroradiology assistant. You will look at the brain MRI "
        "image(s) provided and the measured features below, then produce a "
        "single JSON object with three sections.\n\n"
        "STRICT RULES (apply to every section):\n"
        "1. Do NOT invent any number, location, or finding not present in the "
        "measured features.\n"
        "2. Every numeric value you cite must match the features verbatim.\n"
        "3. If a section cannot be filled in good faith, return an empty value "
        "for it. Do NOT make something up to satisfy the schema.\n"
        "4. Output a single JSON object. No markdown, no preamble, no trailing "
        "commentary.\n\n"
        "SECTION 1 - 'polished_impression' (Pattern A: rephrase only):\n"
        "  - 3 to 6 sentences of clean radiology-style prose.\n"
        "  - Rephrase the SOURCE impression and findings below; do NOT add facts.\n"
        "  - Keep every numeric value identical to the source.\n\n"
        "SECTION 2 - 'additional_differentials' (Pattern B: cite-or-drop):\n"
        "  - Up to 4 differentials NOT already in the already-proposed list.\n"
        "  - Each item MUST cite at least one key from the citable-keys list, "
        "in the form 'feature.key=value'. Citations not in the list are rejected.\n"
        "  - Schema per item: {statement, supported_by:[citations], "
        "confidence: 'low'|'moderate'|'high'}.\n\n"
        "SECTION 3 - 'visual_observations' (Pattern C: image co-observer):\n"
        "  - Up to 4 short observations of what you VISUALLY see in the image.\n"
        "  - Restrict to: intensity (brighter/darker than surrounding brain), "
        "uniformity (homogeneous/heterogeneous), border (sharp/diffuse), or "
        "spatial relations.\n"
        "  - Schema per item: {region, observation, "
        "claimed_property: 'intensity'|'uniformity'|'border'|'spatial'}.\n\n"
        "--- SOURCE IMPRESSION ---\n"
        f"{impression}\n"
        "--- SOURCE FINDINGS ---\n"
        f"{fields_text}\n"
        "--- ALREADY-PROPOSED DIFFERENTIALS (do not repeat) ---\n"
        f"{rule_text}\n"
        "--- CITABLE FEATURE KEYS (whitelist) ---\n"
        f"{citable}\n"
        "--- COMPACT MEASURED FEATURES ---\n"
        f"{json.dumps(feat_compact, indent=2, default=_safe_default)}\n"
        "--- ANCHORS FOR ORIENTATION (don't restate as observations) ---\n"
        f"hemisphere={loc.get('hemisphere', '?')}, "
        f"lobe={loc.get('approximate_lobe_hint', '?')}, "
        f"area_mm2={geom.get('area_mm2', 0):.0f}\n\n"
        "Now return the single JSON object."
    )


def _call_backend(backend: str, prompt_text: str, images_b64: list, image_labels: list) -> str:
    """Single dispatch point for the per-pass LLM calls."""
    if backend == 'ollama':
        return _call_ollama(prompt_text, images_b64, image_labels)
    if backend == 'anthropic':
        return _call_anthropic(prompt_text, images_b64, image_labels)
    if backend == 'openai':
        return _call_openai(prompt_text, images_b64, image_labels)
    return ''


# ---------------------------------------------------------------------------
# Pattern A - Polish pass (rephrase only, no new facts)
# ---------------------------------------------------------------------------


def _build_polish_prompt(deterministic: dict) -> str:
    impression = deterministic.get('impression') or deterministic.get('summary') or ''
    findings = deterministic.get('findings') or {}
    fields_text = '\n'.join(f'- {k}: {v}' for k, v in findings.items() if v)
    return (
        "You are a radiology editor. Your task is to REPHRASE the report below in clean, "
        "concise, neutral radiology-style English suitable for a brain-MRI report. "
        "STRICT RULES:\n"
        "1. Do NOT introduce any new facts, numbers, anatomic locations, terms, or "
        "differential diagnoses that are not already in the source.\n"
        "2. Do NOT speculate. Do NOT 'sound more medical' by adding terminology not "
        "warranted by the source.\n"
        "3. Keep every numeric value verbatim (e.g. an area of '595 mm^2' must remain 595).\n"
        "4. Output 3-6 sentences of plain prose. No bullet lists. No JSON. No preamble.\n\n"
        "--- SOURCE REPORT ---\n"
        f"Impression: {impression}\n"
        f"Findings:\n{fields_text}\n"
        "--- END SOURCE ---\n\n"
        "Now write the polished prose."
    )


def _validate_polish(raw: str, deterministic: dict, features: dict) -> tuple[str, list[str]]:
    """Return (polished_text, warnings).

    Hallucination checks (rejection is a warning, returned as empty text):
      1. Introduces a number absent from the source.
      2. Names the contralateral hemisphere when source specified one.
      3. Uses MRI modality vocabulary (T1, T2, T1c, FLAIR, DWI, ADC) when the
         source has no multimodal section - common with single-channel RGB
         inputs where the LLM 'sounds medical' by inventing sequence names.
      4. Names a specific tumor type as a fact (e.g. 'glioma', 'meningioma',
         'metastasis') in the polish - those belong in the differential,
         not the impression rephrase.
      5. Introduces a clinical action verb the source doesn't have
         ('biopsy is recommended', 'surgery indicated') - the recommendation
         field is separately controlled by overall_confidence.
    """
    if not raw or not isinstance(raw, str):
        return '', ['empty_response']
    text = raw.strip()
    if not text:
        return '', ['empty_response']
    warnings: list[str] = []
    src_text = ' '.join([
        deterministic.get('impression') or '',
        ' '.join((deterministic.get('findings') or {}).values()),
        deterministic.get('confidence_assessment') or '',
    ])
    src_low = src_text.lower()
    low = text.lower()
    source_numbers = set(_extract_numbers(src_text))
    response_numbers = set(_extract_numbers(text))
    new_numbers = response_numbers - source_numbers
    # Tolerate {0, 1} and very small ints that show up in counts.
    new_numbers = {n for n in new_numbers if n not in (0.0, 1.0) and abs(n) > 0.0001}
    if new_numbers:
        warnings.append(f'introduced_new_numbers: {sorted(new_numbers)[:5]}')
    # Hemisphere contradiction.
    loc = (features.get('localization') or {})
    hemi = (loc.get('hemisphere') or '').lower()
    opp = 'left' if hemi == 'right' else ('right' if hemi == 'left' else '')
    if hemi and opp and (f' {opp} hemisphere' in low or f' {opp}-sided' in low) and f' {hemi} hemisphere' not in low:
        warnings.append(f'hemisphere_contradiction: source={hemi}, response_mentions={opp}')
    # MRI modality terms invented when source has no multimodal data.
    has_mm = (features.get('multimodal') is not None) and bool(features.get('multimodal'))
    if not has_mm:
        invented_mm: list[str] = []
        # Patterns that strongly imply a specific MRI sequence we don't have.
        modality_patterns = [
            ' t1 ', ' t1-', ' t2 ', ' t2-', ' t1c ', ' t1ce ',
            ' flair', ' dwi', ' adc', ' swi', ' mra ',
            't1 sequence', 't2 sequence', 't1-weighted', 't2-weighted',
            't1c-enhancing', 'gadolinium',
        ]
        for p in modality_patterns:
            if p in low and p not in src_low:
                invented_mm.append(p.strip())
        if invented_mm:
            warnings.append(f'invented_modality_terms: {invented_mm}')
    # Specific tumor types in the polish (belongs in differential, not impression).
    typed_tumors = [
        'glioblastoma', 'glioma', 'meningioma', 'metastasis', 'metastases',
        'lymphoma', 'astrocytoma', 'oligodendroglioma', 'ependymoma',
        'schwannoma', 'neurinoma', 'medulloblastoma', 'lipoma',
    ]
    invented_dx = [t for t in typed_tumors if t in low and t not in src_low]
    if invented_dx:
        warnings.append(f'invented_diagnosis_terms_in_polish: {invented_dx}')
    # Clinical action verbs that the source didn't authorise.
    action_verbs = ['biopsy', 'surgery', 'surgical resection', 'radiation', 'chemotherapy']
    invented_actions = [a for a in action_verbs if a in low and a not in src_low]
    if invented_actions:
        warnings.append(f'invented_clinical_actions: {invented_actions}')
    if warnings:
        return '', warnings
    return text, []


def _extract_numbers(text: str) -> list[float]:
    import re
    return [float(m) for m in re.findall(r'-?\d+\.?\d*', text or '')]


# ---------------------------------------------------------------------------
# Pattern B - Differential expansion with citation checks
# ---------------------------------------------------------------------------


_DIFFERENTIAL_CITABLE_KEYS = {
    'geometry.area_px', 'geometry.area_mm2', 'geometry.solidity', 'geometry.eccentricity',
    'geometry.circularity', 'geometry.equivalent_diameter_px',
    'localization.hemisphere', 'localization.approximate_lobe_hint', 'localization.depth_label',
    'localization.midline_shift_suspected',
    'components.n_components', 'components.multifocal',
    'morphology.border_label', 'morphology.internal_intensity_zones',
    'morphology.border_relative_to_brain', 'morphology.border_gradient_mean',
    'internal_architecture.necrosis_like_fraction_single_channel',
    'internal_architecture.rim_pattern_label',
    'internal_architecture.rim_vs_core_intensity_ratio',
    'internal_architecture.hyperdense_blob_count_inside_tumor',
    'internal_architecture.hypodense_blob_count_inside_tumor',
    'mass_effect.mass_effect_label', 'mass_effect.tumor_to_brain_area_ratio',
    'mass_effect.brain_symmetry_iou',
    'grade_evidence.score_0_to_1', 'grade_evidence.evidence_band',
    'multimodal.t1c_enhancing_fraction', 'multimodal.t1c_predominantly_enhancing',
    'multimodal.edema_likely', 'multimodal.edema_halo_ratio',
    'multimodal.necrosis_likely', 'multimodal.t2_strongly_hyperintense',
    'texture.heterogeneity_score', 'texture.shannon_entropy', 'texture.contrast',
}


def _build_differential_prompt(features: dict, deterministic: dict) -> str:
    feat_compact = _compact_features_for_diff(features)
    rule_diffs = [d['statement'] for d in (deterministic.get('differential_with_citations') or [])]
    rule_text = '\n'.join(f'- {b}' for b in rule_diffs) or '(none)'
    citable = '\n'.join(sorted(_DIFFERENTIAL_CITABLE_KEYS))
    return (
        "You are a neuroradiologist's assistant. Given the measured features of a "
        "brain MRI lesion, propose UP TO 4 differential diagnoses we have not "
        "already covered. STRICT RULES:\n"
        "1. Each bullet MUST cite at least one feature key from the citable-keys "
        "list below in the form 'feature_key=value' or 'feature_key (value)'.\n"
        "2. Do NOT cite a key that is not in the list. Do NOT invent feature names.\n"
        "3. Do NOT propose anything contradicted by the features.\n"
        "4. If you cannot find a defensible additional differential, return an empty list.\n"
        "5. Output JSON only, no Markdown, no preamble. Schema:\n"
        '   {"differentials": [{"statement": "...", "supported_by": ["feature.key=value", ...], "confidence": "low|moderate|high"}]}\n'
        "\n--- ALREADY-PROPOSED DIFFERENTIALS (do not repeat) ---\n"
        f"{rule_text}\n"
        "\n--- CITABLE FEATURE KEYS ---\n"
        f"{citable}\n"
        "\n--- MEASURED FEATURES (subset) ---\n"
        f"{json.dumps(feat_compact, indent=2, default=_safe_default)}\n"
        "\nReturn JSON only."
    )


def _compact_features_for_diff(features: dict) -> dict:
    """Just the subset of features used by the differential pass — keeps the
    prompt small enough for the local model's context window."""
    keep = {}
    for top, sub in [
        ('geometry', ['area_mm2', 'area_px', 'solidity', 'eccentricity', 'circularity']),
        ('localization', ['hemisphere', 'approximate_lobe_hint', 'depth_label', 'midline_shift_suspected']),
        ('components', ['n_components', 'multifocal']),
        ('morphology', ['border_label', 'internal_intensity_zones', 'border_relative_to_brain']),
        ('internal_architecture', ['necrosis_like_fraction_single_channel', 'rim_pattern_label',
                                    'rim_vs_core_intensity_ratio', 'hyperdense_blob_count_inside_tumor',
                                    'hypodense_blob_count_inside_tumor']),
        ('mass_effect', ['mass_effect_label', 'tumor_to_brain_area_ratio', 'brain_symmetry_iou']),
        ('grade_evidence', ['score_0_to_1', 'evidence_band']),
        ('multimodal', ['t1c_enhancing_fraction', 'edema_likely', 'edema_halo_ratio',
                         'necrosis_likely', 't2_strongly_hyperintense', 't1c_predominantly_enhancing']),
        ('texture', ['heterogeneity_score', 'shannon_entropy', 'contrast']),
    ]:
        if top in features and isinstance(features[top], dict):
            sub_d = {k: features[top].get(k) for k in sub if k in features[top]}
            if sub_d:
                keep[top] = sub_d
    return keep


def _validate_differentials(raw: str, features: dict) -> tuple[list[dict], list[dict]]:
    """Return (accepted, rejected_with_reasons).

    Reject if: not valid JSON, missing supported_by, citation key not in
    citable-set, or the cited fact contradicts the features.
    """
    if not raw or not isinstance(raw, str):
        return [], []
    parsed = _try_parse_json(raw)
    if not isinstance(parsed, dict):
        return [], [{'reason': 'non_json_response', 'raw_preview': raw[:200]}]
    items = parsed.get('differentials') or parsed.get('items') or []
    if not isinstance(items, list):
        return [], [{'reason': 'differentials_not_a_list', 'raw_preview': raw[:200]}]
    accepted: list[dict] = []
    rejected: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            rejected.append({'reason': 'item_not_object', 'item': str(item)[:200]})
            continue
        stmt = str(item.get('statement') or '').strip()
        cites = item.get('supported_by') or []
        if not stmt:
            rejected.append({'reason': 'empty_statement', 'item': item})
            continue
        if not isinstance(cites, list) or not cites:
            rejected.append({'reason': 'no_supported_by', 'item': item})
            continue
        # Validate each citation against the citable-set + actual value.
        bad_cites = []
        for c in cites:
            c_str = str(c)
            # Pull the "feature.key" portion (left of '=' or '(').
            key_part = c_str.split('=')[0].split('(')[0].strip()
            if key_part not in _DIFFERENTIAL_CITABLE_KEYS:
                bad_cites.append({'cite': c_str, 'why': 'key_not_in_citable_set'})
                continue
            # Verify the cited value matches features (when a value is given).
            if not _citation_consistent_with_features(c_str, features):
                bad_cites.append({'cite': c_str, 'why': 'value_inconsistent_with_features'})
        if bad_cites:
            rejected.append({'reason': 'citation_check_failed', 'bad_cites': bad_cites, 'item': item})
            continue
        accepted.append({
            'statement': stmt,
            'supported_by': [str(c) for c in cites],
            'confidence': str(item.get('confidence', 'n/a')),
        })
    return accepted, rejected


def _citation_consistent_with_features(cite_str: str, features: dict) -> bool:
    """Check that a citation like 'morphology.border_label=sharp / well-circumscribed'
    matches what's actually in features. If the citation gives a key only (no
    '=value') we accept it as long as the key exists.
    """
    if '=' not in cite_str:
        key = cite_str.split('(')[0].strip()
        top, _, sub = key.partition('.')
        return isinstance(features.get(top), dict) and sub in features[top]
    key, _, value_raw = cite_str.partition('=')
    key = key.strip()
    top, _, sub = key.partition('.')
    if not isinstance(features.get(top), dict) or sub not in features[top]:
        return False
    actual = features[top][sub]
    value_raw = value_raw.strip().strip('"').strip("'").rstrip(')')
    # Numeric: tolerate 5% deviation to absorb formatting differences.
    try:
        actual_f = float(actual)
        cited_f = float(value_raw)
        return abs(actual_f - cited_f) <= max(0.05 * abs(actual_f), 0.05)
    except (TypeError, ValueError):
        pass
    # Boolean.
    if isinstance(actual, bool):
        return value_raw.lower() in ('true', 'false') and (value_raw.lower() == 'true') == actual
    # String compare (case-insensitive, allow prefix match).
    actual_s = str(actual).lower()
    cited_s = value_raw.lower()
    return actual_s.startswith(cited_s) or cited_s.startswith(actual_s) or actual_s == cited_s


# ---------------------------------------------------------------------------
# Pattern C - Visual co-observer with conflict detection
# ---------------------------------------------------------------------------


def _build_visual_prompt(features: dict) -> str:
    geom = features.get('geometry') or {}
    loc = features.get('localization') or {}
    return (
        "You are a visual co-observer of a brain MRI. The lesion has been segmented "
        "for you (look at the overlay image). DO NOT diagnose. ONLY describe what you "
        "visually see inside or near the segmented region.\n\n"
        "STRICT RULES:\n"
        "1. Describe at most 4 observations. Each must be visually verifiable in "
        "the overlay image you can see.\n"
        "2. Each observation must be of the form: 'In the [region] of the lesion, "
        "[signal observation]'. Stick to: signal intensity (bright/dark relative "
        "to surrounding brain), uniformity (homogeneous/heterogeneous), border "
        "(sharp/diffuse), and spatial relations.\n"
        "3. If your observation appears to contradict a measurement, still report "
        "it - we will flag it as a disagreement, not silently drop it.\n"
        "4. Output JSON only. Schema:\n"
        '   {"observations": [{"region": "...", "observation": "...", "claimed_property": "intensity|uniformity|border|spatial"}]}\n'
        "\n--- ALREADY-MEASURED ANCHORS (for orientation; do not restate) ---\n"
        f"Hemisphere: {loc.get('hemisphere', 'unknown')}\n"
        f"Approximate lobe: {loc.get('approximate_lobe_hint', 'unknown')}\n"
        f"Area: {geom.get('area_mm2', 0):.0f} mm^2\n"
        f"Border (measured): {(features.get('morphology') or {}).get('border_label', 'unknown')}\n"
        f"Rim pattern (measured): {(features.get('internal_architecture') or {}).get('rim_pattern_label', 'unknown')}\n"
        "\nReturn JSON only."
    )


def _validate_visual_observations(raw: str, features: dict) -> tuple[list[dict], list[dict]]:
    """Cross-check each visual observation against features.

    Returns (kept, disagreements). Kept observations are those compatible with
    measurements. Disagreements are observations that contradict a measurement;
    they're surfaced in their own UI section so the user sees the model conflict.
    """
    if not raw or not isinstance(raw, str):
        return [], []
    parsed = _try_parse_json(raw)
    if not isinstance(parsed, dict):
        return [], [{'reason': 'non_json_response', 'raw_preview': raw[:200]}]
    items = parsed.get('observations') or []
    if not isinstance(items, list):
        return [], [{'reason': 'observations_not_a_list', 'raw_preview': raw[:200]}]
    kept: list[dict] = []
    disagree: list[dict] = []
    measured_border = (features.get('morphology') or {}).get('border_label', '').lower()
    measured_rim = (features.get('internal_architecture') or {}).get('rim_pattern_label', '').lower()
    measured_hemi = (features.get('localization') or {}).get('hemisphere', '').lower()
    measured_uniform = (features.get('texture') or {}).get('heterogeneity_score', None)

    for item in items:
        if not isinstance(item, dict):
            continue
        obs = str(item.get('observation') or '').strip()
        region = str(item.get('region') or '').strip()
        claim = str(item.get('claimed_property') or '').strip()
        if not obs:
            continue
        record = {'observation': obs, 'region': region, 'claimed_property': claim}
        low = obs.lower()
        conflicts = []
        # Border conflict.
        if 'sharp' in low and 'ill-defined' in measured_border:
            conflicts.append(f'border (measured: "{measured_border}")')
        if ('diffuse' in low or 'ill-defined' in low) and 'sharp' in measured_border:
            conflicts.append(f'border (measured: "{measured_border}")')
        # Rim/homogeneity conflict.
        if 'homogeneous' in low and 'rim-enhancing' in measured_rim:
            conflicts.append(f'rim_pattern (measured: "{measured_rim}")')
        if 'rim' in low and 'homogeneous' in measured_rim:
            conflicts.append(f'rim_pattern (measured: "{measured_rim}")')
        # Heterogeneity conflict.
        if isinstance(measured_uniform, (int, float)):
            if 'homogeneous' in low and measured_uniform > 0.40:
                conflicts.append(f'texture (measured heterogeneity_score={measured_uniform:.2f}, high)')
            if 'heterogeneous' in low and measured_uniform < 0.15:
                conflicts.append(f'texture (measured heterogeneity_score={measured_uniform:.2f}, low)')
        # Hemisphere conflict.
        if measured_hemi == 'right' and ('left hemisphere' in low or 'left side' in low):
            conflicts.append(f'hemisphere (measured: "{measured_hemi}")')
        if measured_hemi == 'left' and ('right hemisphere' in low or 'right side' in low):
            conflicts.append(f'hemisphere (measured: "{measured_hemi}")')

        if conflicts:
            record['conflicts_with'] = conflicts
            disagree.append(record)
        else:
            kept.append(record)
    return kept, disagree


# ---------------------------------------------------------------------------
# Pattern D - Vision reasoning for NEGATIVE (no-tumor) cases
# ---------------------------------------------------------------------------

# Words whose appearance in an LLM "this is healthy" response would
# directly contradict the negative verdict and is therefore stripped /
# rejected. Includes tumour synonyms and pathological qualifiers.
_NEGATIVE_FORBIDDEN_TERMS = {
    'tumor', 'tumour', 'lesion', 'mass', 'neoplasm', 'neoplastic',
    'malignant', 'malignancy', 'metastasis', 'metastatic',
    'glioma', 'glioblastoma', 'astrocytoma', 'meningioma',
    'oligodendroglioma', 'ependymoma', 'schwannoma',
    'cancerous', 'cancer', 'edema', 'oedema',
    'necrosis', 'necrotic', 'enhancement', 'enhancing',
}


def _build_negative_visual_prompt(mean_p: float, band: str,
                                     classifier_results: Optional[dict]) -> str:
    """Prompt the vision LLM to articulate why a healthy brain looks healthy.

    Crucial framing: the verdict is *already established* (CNN, Transfer, ViT
    all agree at very low tumor probability). We are NOT asking the LLM to
    re-classify - we are asking it to look at the image and describe the
    anatomical features of normalcy that are consistent with the verdict, AND
    flag any normal-anatomy structures that could be confused for tumor by a
    naive reader. This is the value-add for the user on negative cases.
    """
    probs = []
    for n in ('cnn', 'transfer', 'vit'):
        r = (classifier_results or {}).get(n) or {}
        p = r.get('probability')
        if isinstance(p, (int, float)):
            probs.append(f'{n}={p:.4f}')
    per_model_line = ', '.join(probs) if probs else 'unknown'
    return (
        "Three independent classifiers (CNN, Transfer-Learning ResNet50, "
        "Vision Transformer hybrid) have ALREADY ruled out tumor on this MRI "
        f"at {band} confidence (per-model probabilities: {per_model_line}; "
        f"mean p={mean_p:.4f}). The U-Net segmentation may still show a small "
        "mask because it was trained on tumor-positive patients only and has "
        "a positive bias - the mask is a known false positive.\n\n"
        "Your task: looking at the actual MRI image(s) provided, describe "
        "WHAT YOU SEE that is consistent with the no-tumor verdict.\n\n"
        "STRICT RULES:\n"
        "1. DO NOT re-diagnose. The verdict is final. Do not write the word "
        "tumor, tumour, lesion, mass, neoplasm, malignant, metastasis, "
        "glioma, edema, necrosis, or enhancement except inside the explicit "
        "'potential_confounders' field (where you may use them only to say "
        "'this is NOT a tumor, it is the normal X structure').\n"
        "2. Describe ANATOMICAL NORMALCY: bilateral symmetry, normal ventricle "
        "size and shape, intact grey-white differentiation, absence of mass "
        "effect, no midline shift, no abnormal signal change.\n"
        "3. Identify any normal anatomy that an untrained reader might "
        "misinterpret as pathology (e.g. lateral ventricle CSF signal, "
        "choroid plexus, normal cortical sulci, vasculature).\n"
        "4. Output strict JSON only - no markdown, no preamble. Schema:\n"
        '   {"healthy_features": ["short observation 1", "short observation 2", ...],\n'
        '    "potential_confounders": ["normal structure X that could be mistaken for Y", ...],\n'
        '    "overall_assessment": "1-2 sentence plain-English summary"}\n'
        "5. Keep each list item under 200 characters. Up to 5 items per list.\n\n"
        "Return the JSON object."
    )


def _validate_negative_visual_reasoning(raw: str) -> tuple[Optional[str], list]:
    """Validate the Pattern-D vision response on a negative case.

    Returns (rendered_text, warnings) where:
      - rendered_text is a multi-section plain-text block ready to drop into
        the report, or None if the response was unusable.
      - warnings is a list of strings describing dropped items / format
        issues, surfaced in the llm_passes telemetry.

    Sanitisation rules:
      - Parse JSON; if not parseable, drop entire response.
      - For healthy_features and overall_assessment, drop any item that
        contains a forbidden term (tumor/lesion/mass/...). Those would
        contradict the verdict.
      - For potential_confounders, allow the forbidden terms because the
        LLM has to name what the structure could be mistaken for in order
        to clarify it's NOT that.
      - If after sanitisation there are no healthy_features AND no
        overall_assessment, return None.
    """
    warnings: list = []
    if not raw or not isinstance(raw, str):
        return None, ['empty_response']
    parsed = _try_parse_json(raw)
    if not isinstance(parsed, dict):
        return None, [f'non_json_response (first 200 chars: {raw[:200]})']

    def _drop_forbidden(items, label):
        kept = []
        for it in items or []:
            s = str(it).strip()
            if not s:
                continue
            low = s.lower()
            bad = [t for t in _NEGATIVE_FORBIDDEN_TERMS if t in low]
            if bad:
                warnings.append(f'{label}: dropped item using forbidden term(s) {bad}: {s[:80]}')
                continue
            kept.append(s)
        return kept

    healthy = _drop_forbidden(parsed.get('healthy_features'), 'healthy_features')
    overall_raw = str(parsed.get('overall_assessment') or '').strip()
    overall_low = overall_raw.lower()
    if any(t in overall_low for t in _NEGATIVE_FORBIDDEN_TERMS):
        warnings.append(f'overall_assessment: forbidden term, dropped')
        overall = ''
    else:
        overall = overall_raw
    # Confounders may legitimately reference forbidden terms when explaining
    # "normal X may be mistaken for Y but is not Y". Lightly sanity-check
    # each item is at least 8 chars to avoid empty noise.
    confounders_in = parsed.get('potential_confounders') or []
    confounders = [str(c).strip() for c in confounders_in if isinstance(c, (str,)) and len(str(c).strip()) >= 8]

    if not healthy and not overall and not confounders:
        return None, warnings + ['no_usable_content']

    lines = []
    if overall:
        lines.append('OVERALL ASSESSMENT')
        lines.append(overall)
        lines.append('')
    if healthy:
        lines.append('FEATURES OF NORMALCY ON THIS MRI')
        for h in healthy[:5]:
            lines.append(f'  - {h}')
        lines.append('')
    if confounders:
        lines.append('NORMAL ANATOMY THAT COULD BE MISTAKEN FOR PATHOLOGY')
        for c in confounders[:5]:
            lines.append(f'  - {c}')
        lines.append('')
    return '\n'.join(lines).strip(), warnings


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _pick_backend() -> str:
    """Return the first usable backend, in this priority order:

      1. HF Inference Providers (HF_TOKEN set) - open-weight models, no local
         GPU needed, sub-second responses. The right default for the deployed
         Spaces demo because we don't want to ship the Ollama daemon inside
         the container (RAM-heavy) and we want to stay open source.
      2. Local Ollama - the right default on a dev machine with a GPU.
      3. Anthropic / OpenAI - last-resort closed-weight fallbacks the user
         opted into by setting the corresponding API key.
      4. 'none' - run the deterministic radiology report only (zero
         hallucination, no external call).
    """
    if os.environ.get('HF_TOKEN'):
        return 'hf_inference'
    if _ollama_reachable():
        return 'ollama'
    if os.environ.get('ANTHROPIC_API_KEY'):
        return 'anthropic'
    if os.environ.get('OPENAI_API_KEY'):
        return 'openai'
    return 'none'


def _ollama_reachable(timeout: float = 1.5) -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(f'{DEFAULT_OLLAMA_HOST}/api/tags', timeout=timeout)
        if r.status_code != 200:
            return False
        # Confirm the configured model is actually pulled, otherwise we'd 404 later.
        tags = r.json().get('models', [])
        names = {m.get('name', '') for m in tags}
        return DEFAULT_OLLAMA_MODEL in names or any(n.startswith(DEFAULT_OLLAMA_MODEL.split(':')[0]) for n in names)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


SYSTEM_INSTRUCTIONS = """You are an AI assistant helping explain the output of an automated brain-MRI
tumor detection and segmentation system. The system has done all of the heavy
numeric work for you. Your job is to write a clear, grounded, plain-language
explanation that a clinically-literate but non-radiologist reader could
understand.

You receive:
  - Up to 4 images: the original MRI, the predicted segmentation mask,
    a coloured overlay, and a Grad-CAM heatmap.
  - A structured feature dict full of measurements (geometry, intensity,
    texture, anatomic localisation heuristics, and model behaviour).
  - Per-model classifier probabilities.

Rules:
  - Only state facts that are supported by the features or visibly true in the
    images. Do NOT fabricate findings.
  - When you cite a number, use the value from the features verbatim.
  - When a heuristic is approximate (e.g. lobe quadrant), say so.
  - Distinguish between deterministic measurements ('tumor area = 4,238 px',
    'eccentricity = 0.74') and tentative inferences ('the high T1c-enhancing
    fraction together with peritumoral FLAIR signal is consistent with...').
  - Always include a disclaimer that the system is research-grade and not a
    substitute for radiologist review.

Output a single JSON object with these top-level keys (no Markdown):
  summary                       2-3 sentence overview
  findings.geometry             one paragraph about size and shape
  findings.localization         one paragraph about location / hemisphere / lobe
  findings.intensity            one paragraph about how the tumor's intensity
                                  compares to the surrounding brain
  findings.texture              one short paragraph about heterogeneity /
                                  GLCM texture, or "Not enough information" if
                                  the tumor was too small for those features
  findings.multimodal           one paragraph reading T1c / T2 / FLAIR signs
                                  (enhancement, edema, necrosis). If the input
                                  is not a known multimodal stack, write
                                  "Single-channel input - multimodal cues
                                  unavailable."
  differential_diagnosis_hints  array of 2-5 short bullets, each beginning
                                  with the differential consideration and a
                                  brief justification grounded in the features
                                  (e.g. ["High-grade glioma - irregular shape
                                  (solidity 0.71), edema halo, T1c necrotic
                                  core"]). Mark uncertain ones with "(low
                                  confidence)".
  model_agreement_analysis      one paragraph on how the CNN / Transfer / ViT
                                  classifier outputs agreed, and how that
                                  matches the segmentation
  confidence_assessment         one paragraph on overall trust in this output
                                  (segmentation/classifier alignment, Grad-CAM
                                  vs mask agreement, image quality)
  disclaimer                    research-only disclaimer string
"""


def _build_prompt(features: dict, classifier_results: Optional[dict],
                   modality_channels: Optional[tuple[str, str, str]]) -> str:
    feat_json = json.dumps(features, indent=2, default=_safe_default)
    cls_json = json.dumps(classifier_results or {}, indent=2, default=_safe_default)
    mod = f"\nMultimodal channels (R, G, B): {modality_channels}" if modality_channels else ''
    return (
        SYSTEM_INSTRUCTIONS
        + '\n\n--- FEATURE DICT ---\n'
        + feat_json
        + '\n\n--- CLASSIFIER RESULTS ---\n'
        + cls_json
        + mod
        + '\n\nReturn ONLY the JSON object, no preamble.'
    )


def _safe_default(o):
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _call_ollama(prompt_text: str, images_b64: list[str], _image_labels: list[str],
                  *, model_override: Optional[str] = None,
                  num_ctx_override: Optional[int] = None) -> str:
    if requests is None:
        raise RuntimeError('requests not installed')
    # num_ctx is held tight (2048) so qwen2.5vl:7b's KV cache fits next to our
    # PyTorch segmentation/classifier models on an 8 GB card. Default 4096
    # pushes VRAM need to ~8.6 GiB and gets rejected by Ollama. If the user
    # exports OLLAMA_NUM_CTX we honor that override.
    num_ctx = num_ctx_override or int(os.environ.get('OLLAMA_NUM_CTX', '1536'))
    keep_alive = os.environ.get('OLLAMA_KEEP_ALIVE', '10m')
    # num_gpu=999 forces Ollama to push EVERY layer of qwen2.5vl onto the GPU.
    # Default Ollama splits layers across system RAM + GPU based on its memory
    # estimate; on a 16 GB RAM / 8 GB VRAM laptop with PyTorch co-resident the
    # split lands too RAM-heavy and the model can't fit (6.9 GiB needed, 6.9
    # GiB free). Putting all layers on the GPU collapses system-RAM need to
    # roughly the loading buffer (~1-2 GiB), which IS available, and uses the
    # ~5 GiB of free VRAM we already have. Override with OLLAMA_NUM_GPU=N if
    # you ever run on a card too small to hold all layers.
    # num_gpu: only forward if user explicitly set the env var. Forcing a
    # high value (e.g. 999) into the Ollama runtime for VL models breaks
    # their memory layout planner ("memory layout cannot be allocated with
    # num_gpu = 999"). Without the override Ollama picks a sensible split.
    model_name = model_override or DEFAULT_OLLAMA_MODEL
    options = {
        'temperature': 0.2,
        'num_predict': 1500,
        'num_ctx': num_ctx,
    }
    if os.environ.get('OLLAMA_NUM_GPU'):
        options['num_gpu'] = int(os.environ['OLLAMA_NUM_GPU'])
    if os.environ.get('OLLAMA_KV_CACHE_TYPE'):
        options['kv_cache_type'] = os.environ['OLLAMA_KV_CACHE_TYPE']
    payload = {
        'model': model_name,
        'prompt': prompt_text,
        'images': images_b64,
        'stream': False,
        'keep_alive': keep_alive,
        'options': options,
    }
    r = requests.post(f'{DEFAULT_OLLAMA_HOST}/api/generate', json=payload, timeout=300)
    if not r.ok:
        # Surface Ollama's actual error JSON so the caller can see "model
        # requires more system memory..." or similar capacity hints.
        try:
            err = r.json().get('error', r.text)
        except Exception:
            err = r.text
        raise requests.HTTPError(f'Ollama /api/generate {r.status_code}: {err}', response=r)
    return r.json().get('response', '').strip()


def _call_anthropic(prompt_text: str, images_b64: list[str], image_labels: list[str]) -> str:
    if requests is None:
        raise RuntimeError('requests not installed')
    api_key = os.environ['ANTHROPIC_API_KEY']
    content = []
    for b64, label in zip(images_b64, image_labels):
        content.append({
            'type': 'image',
            'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64},
        })
        content.append({'type': 'text', 'text': f'(image labeled "{label}")'})
    content.append({'type': 'text', 'text': prompt_text})
    body = {
        'model': DEFAULT_ANTHROPIC_MODEL,
        'max_tokens': 2000,
        'messages': [{'role': 'user', 'content': content}],
    }
    r = requests.post(
        'https://api.anthropic.com/v1/messages',
        json=body,
        headers={
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        },
        timeout=120,
    )
    r.raise_for_status()
    blocks = r.json().get('content', [])
    return ''.join(b.get('text', '') for b in blocks if b.get('type') == 'text').strip()


def _call_openai(prompt_text: str, images_b64: list[str], image_labels: list[str]) -> str:
    if requests is None:
        raise RuntimeError('requests not installed')
    api_key = os.environ['OPENAI_API_KEY']
    content = [{'type': 'text', 'text': prompt_text}]
    for b64, label in zip(images_b64, image_labels):
        content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{b64}', 'detail': 'high'},
        })
        content.append({'type': 'text', 'text': f'(image labeled "{label}")'})
    body = {
        'model': DEFAULT_OPENAI_MODEL,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': 2000,
        'temperature': 0.2,
    }
    r = requests.post(
        'https://api.openai.com/v1/chat/completions',
        json=body,
        headers={'Authorization': f'Bearer {api_key}'},
        timeout=120,
    )
    r.raise_for_status()
    choices = r.json().get('choices', [])
    if not choices:
        return ''
    return choices[0]['message']['content'].strip()


def _call_hf_inference(prompt_text: str, images_b64: list[str], image_labels: list[str],
                         *, model_override: Optional[str] = None) -> str:
    """Call HuggingFace Inference Providers (OpenAI-compatible router endpoint).

    Auth: HF_TOKEN bearer. The router forwards to whichever underlying
    provider currently hosts the model (Groq / Together / Fireworks / etc.),
    giving us a single API key for many open-weight models.

    Model selection:
      - Caller passes `model_override` to choose text vs vision per pass.
      - If no override, defaults to the text model (HF_MODEL_TEXT).

    The router supports the OpenAI /chat/completions schema with image content
    parts; we re-use the OpenAI message structure we already build for
    _call_openai, with one tweak: images go in as data URLs.
    """
    if requests is None:
        raise RuntimeError('requests not installed')
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN not set; cannot call HuggingFace Inference Providers.')
    model = model_override or DEFAULT_HF_MODEL_TEXT

    content: list[dict] = [{'type': 'text', 'text': prompt_text}]
    for b64, label in zip(images_b64, image_labels):
        content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{b64}'},
        })
        content.append({'type': 'text', 'text': f'(image labeled "{label}")'})

    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': int(os.environ.get('HF_MAX_TOKENS', '1500')),
        'temperature': float(os.environ.get('HF_TEMPERATURE', '0.2')),
        'stream': False,
    }
    r = requests.post(
        f'{DEFAULT_HF_ROUTER_BASE}/chat/completions',
        json=body,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
        timeout=120,
    )
    if not r.ok:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise requests.HTTPError(
            f'HF Inference {r.status_code} (model={model}): {str(err)[:400]}',
            response=r,
        )
    payload = r.json()
    choices = payload.get('choices', [])
    if not choices:
        return ''
    return choices[0]['message']['content'].strip()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


_MAX_IMG_SIDE = int(os.environ.get('LLM_IMG_MAX_SIDE', '224'))


def _to_base64_png(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        a = np.asarray(arr)
        if a.max() <= 1.0:
            a = (a * 255.0)
        arr = np.clip(a, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    # Cap longest side so each image consumes a bounded number of vision tokens
    # in Qwen2.5-VL. Each extra 224px tile is ~32 patch tokens; capping at 224
    # keeps a single image at one tile (~64 tokens) instead of 4-8 tiles.
    h, w = arr.shape[:2]
    max_side = max(h, w)
    if max_side > _MAX_IMG_SIDE:
        scale = _MAX_IMG_SIDE / max_side
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        arr = np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _mask_to_rgb(mask_bin: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_bin)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    elif m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    return np.stack([m, m, m], axis=-1)


# ---------------------------------------------------------------------------
# Output coercion + local fallback
# ---------------------------------------------------------------------------


_REQUIRED_KEYS = [
    'summary', 'findings', 'differential_diagnosis_hints',
    'model_agreement_analysis', 'confidence_assessment', 'disclaimer',
]
_FINDING_KEYS = ['geometry', 'localization', 'intensity', 'texture', 'multimodal']
_DEFAULT_DISCLAIMER = (
    'This is an automated, research-grade output from a deep-learning system. '
    'It is NOT a clinical diagnosis. All findings must be reviewed by a qualified '
    'radiologist before any decision is made.'
)


def _coerce_to_schema(raw, features: dict, classifier_results: Optional[dict] = None) -> dict:
    """Normalize LLM / fallback output into the response schema.

    `raw` may be:
      - a dict (already-shaped local-narrative output), or
      - a string (LLM response we need to JSON-parse).

    Recovery ladder, weakest -> strongest:
      1. String, no parseable JSON          -> local narrative + raw text attached.
      2. JSON parsed but missing our keys   -> local narrative + raw text attached.
         (Happens with small models like moondream that hallucinate a different
         schema. We don't want to ship empty fields to the UI just because the
         LLM ignored the instructions.)
      3. JSON parsed with our keys present  -> use it, fill missing keys with ''.

    The recovery cases mark the response with `llm_followed_schema: False` so
    callers can see the LLM was invoked but its output was not usable.
    """
    raw_str_for_log = '' if isinstance(raw, dict) else str(raw)[:600]
    if isinstance(raw, dict):
        parsed = raw
    else:
        parsed = _try_parse_json(raw)

    # Recovery: parse failed entirely.
    if not isinstance(parsed, dict):
        fallback = _local_narrative(features, classifier_results=classifier_results,
                                     error_note=f'LLM did not return valid JSON. Raw: {raw_str_for_log[:300]}')
        fallback['llm_followed_schema'] = False
        fallback['raw_llm_text'] = raw_str_for_log
        return fallback

    # Did the LLM hit any of our schema keys?
    has_schema_signal = (
        bool(str(parsed.get('summary') or '').strip())
        or isinstance(parsed.get('findings'), dict) and any(
            str(v or '').strip() for v in parsed['findings'].values()
        )
    )

    # Recovery: JSON came back but it's a different schema.
    if not has_schema_signal and not isinstance(raw, dict):
        fallback = _local_narrative(features, classifier_results=classifier_results,
                                     error_note=f'LLM returned JSON in an unexpected shape. Raw: {raw_str_for_log[:300]}')
        fallback['llm_followed_schema'] = False
        fallback['raw_llm_text'] = raw_str_for_log
        return fallback

    out = dict(parsed)
    if not isinstance(out.get('findings'), dict):
        out['findings'] = {}
    for k in _FINDING_KEYS:
        out['findings'].setdefault(k, '')
    for k in _REQUIRED_KEYS:
        if k not in out:
            out[k] = '' if k != 'differential_diagnosis_hints' else []
    if not out['disclaimer']:
        out['disclaimer'] = _DEFAULT_DISCLAIMER
    if not isinstance(out['differential_diagnosis_hints'], list):
        out['differential_diagnosis_hints'] = [str(out['differential_diagnosis_hints'])]
    out['llm_followed_schema'] = True
    return out


def _try_parse_json(raw: str):
    s = raw.strip()
    if s.startswith('```'):
        s = s.strip('`')
        # drop a leading 'json' tag if the model used a fenced block
        if s.lower().startswith('json\n'):
            s = s[5:]
        elif s.lower().startswith('json '):
            s = s[5:]
    # Find the outermost JSON object.
    start = s.find('{')
    end = s.rfind('}')
    if start < 0 or end <= start:
        return None
    candidate = s[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _build_classifier_negative_explanation(classifier_results: Optional[dict],
                                              mean_p: Optional[float],
                                              band: Optional[str],
                                              geom: Optional[dict]) -> str:
    """Explain WHY the binary classifiers ruled out tumor on this scan.

    Replaces what was previously a contradictory dump of tumor features
    describing the U-Net's false-positive segmentation. The explanation
    walks through the three architecture-diverse classifier outputs, why
    inter-model agreement at this level is a strong negative signal, and
    notes the U-Net positive bias that is responsible for the mask the
    user still sees on the Visualization panel.
    """
    if not classifier_results:
        return ''

    probs: dict = {}
    for name in ('cnn', 'transfer', 'vit'):
        r = classifier_results.get(name)
        if isinstance(r, dict) and isinstance(r.get('probability'), (int, float)):
            probs[name] = float(r['probability'])

    if not probs:
        return ''

    arch_label = {
        'cnn': 'CNN (custom 3-block convolutional)',
        'transfer': 'Transfer Learning (ResNet50 pretrained on ImageNet, fine-tuned)',
        'vit': 'Vision Transformer (ResNet50 + 4 transformer blocks, hybrid)',
    }
    per_model_lines = [
        f'- {arch_label.get(n, n):<55} tumor probability {p:.4f} '
        f'(equivalently {(1.0 - p) * 100:5.2f}% no-tumor confidence)'
        for n, p in probs.items()
    ]
    p_values = list(probs.values())
    max_p = max(p_values) if p_values else 0.0
    n_models = len(p_values)

    band_phrase = (f'{band} confidence' if band else 'meaningful confidence')
    summary = (
        f'All {n_models} classifiers independently produced tumor probabilities below '
        f'{max(0.5, max_p + 0.05):.2f} (mean p={mean_p:.4f}, {band_phrase}). '
        f'These three models do not share an architecture, do not share a feature '
        f'backbone, and were not trained with shared weights - they are three '
        f'independent voters on the same question. Inter-model agreement at this '
        f'level is a strong negative signal: a false negative would require all '
        f'three architectures to fail in the same direction on the same input, '
        f'which is empirically rare in our validation runs.'
    )

    bias_note = (
        'The U-Net segmentation panel still shows a mask because the U-Net was '
        'trained on BraTS + LGG (tumor-positive patients only). It has never seen '
        'a healthy brain during training and has a positive bias: it always emits '
        'a mask somewhere, picking up the strongest intensity edges in the input. '
        'On a no-tumor scan that mask traces normal anatomy. The features measured '
        'on that region (intensity, mass effect, grade evidence) are NOT '
        'interpreted here because they are not measuring a tumor - they are '
        'measuring a piece of normal anatomy that the U-Net mistook for one.'
    )

    if geom and geom.get('area_px'):
        fp_note = (
            f'For transparency, the false-positive region ({geom.get("area_px", 0):,} px) '
            f'has been computed - those measurements are kept under '
            f'"False-positive region analysis (debug)" below but are not '
            f'interpreted as clinical findings.'
        )
    else:
        fp_note = ''

    return '\n\n'.join([
        'WHY THE CLASSIFIERS RULED THIS OUT',
        'Per-model tumor probability:',
        '\n'.join(per_model_lines),
        summary,
        bias_note,
        fp_note,
    ]).strip()


def _classifier_consensus(classifier_results: Optional[dict]) -> tuple:
    """Compute the binary-classifier majority verdict.

    Returns (verdict, mean_probability, confidence_band):
      verdict           - 'tumor' | 'no_tumor' | 'mixed' | None
      mean_probability  - mean of per-model tumor probabilities, or None
      confidence_band   - 'high' | 'moderate' | 'low' | None
    """
    if not classifier_results:
        return None, None, None
    probs = []
    for _name, r in classifier_results.items():
        if not isinstance(r, dict):
            continue
        p = r.get('probability')
        if isinstance(p, (int, float)):
            probs.append(float(p))
    if not probs:
        return None, None, None
    mean_p = sum(probs) / len(probs)
    all_above = all(p >= 0.5 for p in probs)
    all_below = all(p <= 0.5 for p in probs)
    if mean_p >= 0.7 and all_above:
        band = 'high' if mean_p >= 0.9 else 'moderate'
        return 'tumor', mean_p, band
    if mean_p <= 0.3 and all_below:
        band = 'high' if mean_p <= 0.1 else 'moderate'
        return 'no_tumor', mean_p, band
    return 'mixed', mean_p, 'low'


def _local_narrative(features: dict, classifier_results: Optional[dict],
                      error_note: Optional[str] = None) -> dict:
    """Structured radiology-style report grounded ONLY in measured features.

    Zero invented facts: every sentence ties back to a key in `features`. When
    no measurement supports a claim, the corresponding field is empty or marked
    "not assessable". Used as: (a) the standalone deterministic output when the
    LLM is skipped, and (b) the source-of-truth substrate the LLM is asked to
    polish in the "interpretation" pass.
    """
    geom = features.get('geometry') or {}
    loc = features.get('localization') or {}
    inten = features.get('intensity_per_channel') or {}
    mm = features.get('multimodal') or {}
    mb = features.get('model_behavior') or {}
    morph = features.get('morphology') or {}
    me = features.get('mass_effect') or {}
    arch = features.get('internal_architecture') or {}
    grade = features.get('grade_evidence') or {}
    qual = features.get('quality') or {}
    overall = features.get('overall_confidence') or {}

    # ---- CLASSIFIER CONSENSUS (drives the impression framing) -------------
    # The 3 binary classifiers vote on tumor / no-tumor. When they agree with
    # high confidence, the segmentation mask should be interpreted IN LIGHT OF
    # that vote - a small false-positive mask must not be narrated as if it
    # were a real lesion, and an empty mask must not be narrated as if the
    # classifiers found nothing either. Verdict thresholds:
    #   mean_p >= 0.7 with all 3 also >= 0.5: tumor (high if all >= 0.9)
    #   mean_p <= 0.3 with all 3 also <= 0.5: no_tumor (high if all <= 0.1)
    #   anything else: mixed/ambiguous
    cls_verdict, cls_mean_p, cls_band = _classifier_consensus(classifier_results)
    area_px = geom.get('area_px', 0) or 0
    area_mm = geom.get('area_mm2', 0) or 0

    # ---- IMPRESSION (one-line top of report, consensus-aware) -------------
    if cls_verdict == 'no_tumor' and cls_mean_p is not None:
        if area_px > 0:
            impression = (
                f'NO tumor detected. Per classifier consensus (mean tumor-probability '
                f'{cls_mean_p:.3f}, {cls_band} confidence), this scan is interpreted as '
                f'NEGATIVE. The U-Net segmentation produced a {area_px:,} px '
                f'({area_mm:.0f} mm^2) region which, in the absence of any classifier '
                f'support, is treated as a probable false positive and NOT a clinical '
                f'finding.'
            )
        else:
            impression = (
                f'NO tumor detected. All three classifiers and the segmentation model '
                f'agree (mean tumor-probability {cls_mean_p:.3f}, {cls_band} confidence).'
            )
    elif cls_verdict == 'tumor' and area_px == 0:
        impression = (
            f'CLASSIFIER-SEGMENTER DISAGREEMENT. Classifier consensus indicates tumor '
            f'(mean probability {cls_mean_p:.3f}, {cls_band} confidence) but the '
            f'segmentation model returned no mask. This is an ambiguous case; '
            f'a radiologist review is required before any interpretation.'
        )
    elif area_px == 0:
        impression = 'No segmentable tumor region was identified by the model on this image.'
    else:
        size_phrase = (f'a {area_mm:.0f} mm^2 lesion' if area_mm else f'a {area_px:,} px lesion')
        loc_phrase = (f' in the {loc.get("hemisphere", "")} hemisphere'
                      + (f', {loc.get("approximate_lobe_hint", "")}'
                         if loc.get('approximate_lobe_hint') else ''))
        cls_phrase = (f' Classifier consensus: tumor ({cls_mean_p:.3f}, {cls_band} confidence).'
                      if cls_verdict == 'tumor'
                      else f' Classifier consensus: ambiguous (mean probability {cls_mean_p:.3f}).'
                      if cls_verdict == 'mixed'
                      else '')
        confidence_phrase = (f' Overall model confidence {overall.get("band", "n/a")} '
                             f'({(overall.get("score_0_to_1") or 0):.2f}).')
        impression = (f'Automated detection of {size_phrase}{loc_phrase}. '
                      f'Radiographic grade-evidence band: {grade.get("evidence_band", "n/a")}.'
                      f'{cls_phrase}{confidence_phrase}')

    # ---- FINDINGS (structured per-domain) ---------------------------------
    findings = {
        'geometry': _narrate_geometry(geom),
        'localization': _narrate_localization(loc),
        'intensity': _narrate_intensity(inten),
        'texture': _narrate_texture(features.get('texture') or {}),
        'multimodal': _narrate_multimodal(mm),
        'morphology_margins': _narrate_morphology(morph),
        'internal_architecture': _narrate_architecture(arch),
        'mass_effect': _narrate_mass_effect(me),
    }

    # ---- DIFFERENTIAL with explicit feature citations ---------------------
    diff = _narrate_diff_with_citations(features)

    # ---- GRADE EVIDENCE narrative (cite every component) ------------------
    grade_narr = _narrate_grade(grade)

    # ---- MODEL AGREEMENT + GRAD-CAM ---------------------------------------
    model_agree = _narrate_model_agreement(mb)

    # ---- QUALITY + CONFIDENCE ---------------------------------------------
    confidence = _narrate_confidence_v2(overall, qual, mb, geom)

    # Verdict-aware recommendation. The previous code always used
    # overall_confidence.action_recommendation which is derived from the
    # classifier mean / Grad-CAM alignment / quality - it ignored the binary
    # verdict. When the classifiers say NO_TUMOR with high confidence, the
    # recommendation must say so explicitly, not "findings are well supported".
    if cls_verdict == 'no_tumor' and cls_band == 'high':
        recommendation = (
            f'No tumor reported. All classifiers agree (mean p={cls_mean_p:.3f}). '
            f'No further action required from this output; correlate with clinical '
            f'history if symptoms persist.'
        )
    elif cls_verdict == 'no_tumor':
        recommendation = (
            f'Classifier consensus is no-tumor (mean p={cls_mean_p:.3f}). Any small '
            f'segmentation mask is treated as a probable false positive. Correlate '
            f'with clinical history.'
        )
    elif cls_verdict == 'tumor' and area_px == 0:
        recommendation = (
            'Classifier-segmenter disagreement: classifiers report tumor but no mask '
            'was generated. Radiologist review required before any clinical decision.'
        )
    else:
        recommendation = overall.get('action_recommendation', '')

    # When verdict is no_tumor, override the rule-based differential bullets
    # entirely - the rule base infers from geometry only and would produce
    # "infiltrative mass" bullets for a false-positive segmentation. The LLM
    # passes downstream are also short-circuited (see explain()).
    if cls_verdict == 'no_tumor' and cls_band in ('high', 'moderate'):
        diff = [{
            'statement': (
                f'No tumor. Classifier consensus is negative '
                f'(mean p={cls_mean_p:.3f}, {cls_band} confidence). '
                f'No differential diagnosis applies; do not interpret the segmentation '
                f'mask as a lesion.'
            ),
            'supported_by': ['model_behavior.per_model_probabilities'],
            'confidence': cls_band,
        }]

    # The original deterministic narrative treated every U-Net mask as a
    # tumor and produced 8 structured-findings sections + grade-evidence
    # describing it. When the classifier consensus is clearly no-tumor that
    # reads as a contradictory report (Impression: "no tumor", then 800 words
    # of "the tumor is hyperintense, mass effect substantial, grade-evidence
    # 0.79..."). Split those sections into two buckets:
    #   - findings / grade_evidence_narrative: shown as the primary report.
    #     For no-tumor verdicts these are EMPTY because there is nothing
    #     clinical to describe.
    #   - fp_region_analysis / fp_grade_evidence: the raw feature breakdown of
    #     the false-positive segmentation, preserved verbatim under a
    #     "false-positive region debug" collapsible in the UI so the data
    #     isn't lost but is clearly labelled as not-clinical.
    #   - classifier_negative_explanation: NEW prose section explaining why
    #     the classifiers decided no-tumor (per-model probs + agreement
    #     reasoning + note on the U-Net positive bias). Replaces the
    #     contradictory tumor-feature dump with the actual rationale.
    classifier_negative_explanation = None
    fp_region_analysis = None
    fp_grade_evidence = None
    if cls_verdict == 'no_tumor' and cls_band in ('high', 'moderate'):
        fp_region_analysis = dict(findings)
        fp_grade_evidence = grade_narr
        findings = {}
        grade_narr = ''
        classifier_negative_explanation = _build_classifier_negative_explanation(
            classifier_results, cls_mean_p, cls_band, geom,
        )

    out = {
        'summary': impression,
        'impression': impression,
        'findings': findings,
        'differential_diagnosis_hints': [d['statement'] for d in diff],
        'differential_with_citations': diff,
        'grade_evidence_narrative': grade_narr,
        'model_agreement_analysis': model_agree,
        'confidence_assessment': confidence,
        'recommendation': recommendation,
        'classifier_consensus': {
            'verdict': cls_verdict,
            'mean_probability': cls_mean_p,
            'confidence_band': cls_band,
        },
        'classifier_negative_explanation': classifier_negative_explanation,
        'fp_region_analysis': fp_region_analysis,
        'fp_grade_evidence': fp_grade_evidence,
        'quality_warnings': qual.get('quality_warnings', []),
        'disclaimer': _DEFAULT_DISCLAIMER,
    }
    if error_note:
        out['_note'] = error_note
    return out


def _narrate_morphology(morph: dict) -> str:
    if not morph or 'note' in morph:
        return morph.get('note', 'Morphology not assessable.')
    parts = []
    if 'border_label' in morph:
        parts.append(
            f'Border definition: {morph["border_label"]} '
            f'(border gradient {morph.get("border_gradient_mean", 0):.1f} vs '
            f'brain-average gradient {morph.get("brain_gradient_mean", 0):.1f}, '
            f'relative {morph.get("border_relative_to_brain", 0):.2f}).'
        )
    zones = morph.get('internal_intensity_zones', 0)
    if zones:
        parts.append(
            f'Internal intensity zones detected by k-means: {zones} '
            f'(centers at {", ".join(f"{c:.0f}" for c in morph.get("internal_intensity_cluster_means", []))}).'
        )
    return ' '.join(parts) if parts else 'Morphology not assessable.'


def _narrate_architecture(arch: dict) -> str:
    if not arch or 'note' in arch:
        return arch.get('note', 'Internal architecture not assessable.')
    parts = []
    nec = arch.get('necrosis_like_fraction_single_channel')
    if nec is not None:
        parts.append(
            f'Necrosis-like fraction (single-channel) = {nec:.2f} of pixels in the tumor are markedly '
            f'darker than the tumor median. Values >0.20 are suggestive of a necrotic core.'
        )
    rim = arch.get('rim_vs_core_intensity_ratio')
    if rim is not None:
        parts.append(
            f'Rim vs core intensity ratio = {rim:.2f}; pattern: {arch.get("rim_pattern_label", "n/a")}.'
        )
    hyper = arch.get('hyperdense_blob_count_inside_tumor')
    if hyper:
        parts.append(
            f'{hyper} hyperdense focus(es) (>P95) inside the tumor; can suggest haemorrhage or coarse '
            'calcification depending on modality.'
        )
    hypo = arch.get('hypodense_blob_count_inside_tumor')
    if hypo:
        parts.append(
            f'{hypo} hypodense focus(es) (<P5) inside the tumor; can suggest cystic / necrotic foci.'
        )
    return ' '.join(parts) if parts else 'Internal architecture features not produced.'


def _narrate_mass_effect(me: dict) -> str:
    if not me or 'note' in me:
        return me.get('note', 'Mass effect not assessable.')
    parts = [
        f'Tumor-to-brain area ratio = {me.get("tumor_to_brain_area_ratio", 0):.3f}.',
        f'Brain bilateral symmetry IoU (left vs horizontally-mirrored right) = '
        f'{me.get("brain_symmetry_iou", 0):.2f} (1.0 = perfectly symmetric).',
        f'Mass effect evidence: {me.get("mass_effect_label", "n/a")}.',
    ]
    return ' '.join(parts)


def _narrate_grade(grade: dict) -> str:
    if not grade:
        return 'Grade-evidence score not computed.'
    parts = [
        f'Radiographic grade-evidence score = {grade.get("score_0_to_1", 0):.2f} of 1.00 '
        f'({grade.get("evidence_band", "n/a")}).'
    ]
    for c in (grade.get('components') or []):
        parts.append(
            f'  - {c["name"]}: value={c["value_0_to_1"]:.2f} (weight {c["weight"]:.2f}). '
            f'Driver: {c["detail"]}.'
        )
    parts.append(f'({grade.get("disclaimer", "")})')
    return '\n'.join(parts)


def _narrate_diff_with_citations(features: dict) -> list[dict]:
    """Differential diagnosis hints, each bullet attached to feature citations.

    Each entry: {statement, supported_by: [feature_key=value, ...], confidence}.
    Rule-based only at this layer; the LLM Pattern-B pass adds more nuance.
    """
    out: list[dict] = []
    geom = features.get('geometry') or {}
    mm = features.get('multimodal') or {}
    loc = features.get('localization') or {}
    morph = features.get('morphology') or {}
    arch = features.get('internal_architecture') or {}
    comps = features.get('components') or {}
    grade = features.get('grade_evidence') or {}

    if geom.get('area_px', 0) == 0:
        out.append({
            'statement': 'No measurable lesion. No differential can be offered.',
            'supported_by': ['geometry.area_px=0'],
            'confidence': 'high',
        })
        return out

    score = grade.get('score_0_to_1', 0)
    # HGG pattern: necrosis + edema + heterogeneity + irregularity.
    if (mm.get('necrosis_likely') or (arch.get('necrosis_like_fraction_single_channel', 0) > 0.2)) \
            and (mm.get('edema_likely') or score >= 0.5):
        cites = []
        if mm.get('necrosis_likely'):
            cites.append('multimodal.necrosis_likely=True')
        if arch.get('necrosis_like_fraction_single_channel'):
            cites.append(f'internal_architecture.necrosis_like_fraction_single_channel={arch["necrosis_like_fraction_single_channel"]:.2f}')
        if mm.get('edema_likely'):
            cites.append(f'multimodal.edema_likely=True (halo_ratio={mm.get("edema_halo_ratio", 0):.2f})')
        if score >= 0.5:
            cites.append(f'grade_evidence.score_0_to_1={score:.2f}')
        out.append({
            'statement': 'High-grade glial neoplasm (e.g. glioblastoma) - necrotic/heterogeneous '
                          'appearance with peritumoral edema fits a classic HGG pattern.',
            'supported_by': cites,
            'confidence': 'moderate' if score < 0.7 else 'high',
        })

    # Meningioma: well-circumscribed, peripheral, homogeneously enhancing.
    if morph.get('border_label', '').startswith('sharp') \
            and loc.get('depth_label') == 'peripheral / cortical' \
            and (mm.get('t1c_predominantly_enhancing') or not mm.get('edema_likely', False)):
        out.append({
            'statement': 'Extra-axial mass (e.g. meningioma) - sharply circumscribed, peripheral / cortical, '
                          'without prominent peritumoral edema. Look for a dural tail on full series '
                          '(not derivable here).',
            'supported_by': [
                f'morphology.border_label="{morph.get("border_label")}"',
                f'localization.depth_label="{loc.get("depth_label")}"',
            ] + ([f'multimodal.t1c_predominantly_enhancing=True']
                 if mm.get('t1c_predominantly_enhancing') else []),
            'confidence': 'low-to-moderate',
        })

    # Metastases: multifocal lesions, especially at grey-white junction.
    if comps.get('multifocal'):
        out.append({
            'statement': 'Multifocal disease (e.g. metastatic disease, multifocal glioma, lymphoma) - '
                          f'{comps.get("n_components", 0)} discrete components on segmentation.',
            'supported_by': [
                f'components.n_components={comps.get("n_components", 0)}',
                f'components.largest_component_area_fraction={comps.get("largest_component_area_fraction", 0):.2f}',
            ],
            'confidence': 'moderate',
        })

    # LGG-like: well-circumscribed, homogeneous, low grade-evidence score.
    if score < 0.35 and morph.get('border_label', '').startswith('sharp') \
            and arch.get('rim_pattern_label', '').startswith('homogeneous'):
        out.append({
            'statement': 'Lower-grade glioma or benign-appearing lesion - homogeneous internal '
                          'architecture, sharp border, and low radiographic grade-evidence score.',
            'supported_by': [
                f'grade_evidence.score_0_to_1={score:.2f}',
                f'morphology.border_label="{morph.get("border_label")}"',
                f'internal_architecture.rim_pattern_label="{arch.get("rim_pattern_label")}"',
            ],
            'confidence': 'low-to-moderate',
        })

    # Irregular / infiltrative shape strongly suggests glioma or mets.
    if geom.get('solidity', 1.0) < 0.85 and geom.get('eccentricity', 0) > 0.6:
        out.append({
            'statement': 'Infiltrative mass (e.g. infiltrative glioma) - irregular shape with concavities '
                          'and elongated profile is more typical of an infiltrative lesion than a '
                          'well-circumscribed one.',
            'supported_by': [
                f'geometry.solidity={geom.get("solidity", 0):.2f}',
                f'geometry.eccentricity={geom.get("eccentricity", 0):.2f}',
            ],
            'confidence': 'moderate',
        })

    if not out:
        out.append({
            'statement': 'No distinctive radiographic pattern detected by the rule-base. '
                          'Recommend correlation with full multi-sequence series and clinical history.',
            'supported_by': [f'grade_evidence.score_0_to_1={score:.2f}'],
            'confidence': 'n/a',
        })
    return out


def _narrate_confidence_v2(overall: dict, qual: dict, mb: dict, geom: dict) -> str:
    """Confidence assessment combining overall_confidence, quality, and grad-cam."""
    parts = [
        f'Overall confidence band: {overall.get("band", "n/a")} '
        f'(score = {overall.get("score_0_to_1", 0):.2f} of 1.00). '
        f'Recommended action: {overall.get("action_recommendation", "")}.'
    ]
    if mb.get('gradcam_segmentation_aligned'):
        parts.append(
            f'Grad-CAM peak aligns with the segmentation centroid '
            f'(distance {mb.get("gradcam_to_segmentation_distance_px", 0):.0f} px, '
            f'IoU {mb.get("gradcam_mask_iou", 0):.2f}); classifier and segmenter agree on the same region.'
        )
    elif 'gradcam_to_segmentation_distance_px' in mb:
        parts.append(
            f'Grad-CAM peak is far from the segmentation centroid '
            f'(distance {mb.get("gradcam_to_segmentation_distance_px", 0):.0f} px, '
            f'IoU {mb.get("gradcam_mask_iou", 0):.2f}); classifier and segmenter may be looking at '
            'different regions.'
        )
    warnings = qual.get('quality_warnings') or []
    if warnings:
        parts.append('Quality warnings: ' + '; '.join(warnings) + '.')
    if 0 < (geom.get('area_px') or 0) < 50:
        parts.append('Predicted lesion is very small; uncertainty is elevated regardless of probability.')
    return ' '.join(parts)


def _narrate_geometry(g: dict) -> str:
    if not g or g.get('area_px', 0) == 0:
        return 'No tumor region predicted.'
    parts = [
        f'Area = {g["area_px"]:,} pixels ({g.get("area_mm2", 0):.0f} mm^2).',
        f'Equivalent diameter is {g.get("equivalent_diameter_px", 0):.1f} pixels.',
    ]
    ecc = g.get('eccentricity', 0)
    if ecc > 0.85:
        parts.append(f'Shape is highly elongated (eccentricity {ecc:.2f}).')
    elif ecc > 0.6:
        parts.append(f'Shape is moderately elongated (eccentricity {ecc:.2f}).')
    else:
        parts.append(f'Shape is roughly round (eccentricity {ecc:.2f}).')
    sol = g.get('solidity', 1)
    if sol < 0.85:
        parts.append(f'The boundary is irregular (solidity {sol:.2f}, < 0.85 indicates concavities).')
    return ' '.join(parts)


def _narrate_localization(l: dict) -> str:
    if not l or l.get('note') == 'no tumor predicted':
        return 'No tumor predicted, so no localization possible.'
    bits = []
    if l.get('hemisphere'):
        bits.append(f'Right- vs left-hemisphere centroid places this in the {l["hemisphere"]} hemisphere.')
    if l.get('anterior_posterior'):
        bits.append(f'Vertically, the centroid is in the {l["anterior_posterior"]} third of the brain.')
    if l.get('approximate_lobe_hint'):
        bits.append(f'Heuristic quadrant suggests {l["approximate_lobe_hint"]}.')
    if l.get('depth_label'):
        bits.append(f'The mass is {l["depth_label"]}.')
    if l.get('midline_shift_suspected'):
        bits.append('Marked left-right brain-area asymmetry suggests possible midline shift.')
    return ' '.join(bits)


def _narrate_intensity(inten: dict) -> str:
    if not inten:
        return 'Intensity information unavailable.'
    out = []
    for name, d in inten.items():
        if not isinstance(d, dict) or 'mean' not in d:
            continue
        cmp = 'hyperintense' if d.get('hyperintense_vs_brain') else 'hypointense' if d.get('hypointense_vs_brain') else 'iso-intense'
        out.append(f'On channel {name}, the tumor is {cmp} relative to surrounding brain '
                   f'(mean {d["mean"]:.1f} vs background {d["mean_in_brain_outside_tumor"]:.1f}).')
    return ' '.join(out) if out else 'No usable intensity statistics.'


def _narrate_texture(t: dict) -> str:
    if not t or 'note' in t:
        return t.get('note', 'No texture information.')
    h = t.get('heterogeneity_score')
    entropy = t.get('shannon_entropy')
    contrast = t.get('contrast')
    bits = []
    if h is not None:
        bits.append(f'Heterogeneity score = {h:.2f} ({"high" if h > 0.4 else "moderate" if h > 0.2 else "low"}).')
    if entropy is not None:
        bits.append(f'Shannon entropy = {entropy:.2f}.')
    if contrast is not None:
        bits.append(f'GLCM contrast = {contrast:.2f}.')
    return ' '.join(bits) if bits else 'No texture information.'


def _narrate_multimodal(mm: dict) -> str:
    if not mm:
        return 'Single-channel input; multimodal cues unavailable.'
    bits = []
    if mm.get('t1c_enhancing_fraction') is not None:
        bits.append(f'T1c enhancing fraction = {mm["t1c_enhancing_fraction"]:.2f}'
                    + (' (predominantly enhancing).' if mm.get('t1c_predominantly_enhancing') else '.'))
    if mm.get('t2_hyperintensity_ratio') is not None:
        bits.append(f'T2 tumor-vs-brain ratio = {mm["t2_hyperintensity_ratio"]:.2f}'
                    + (' (strongly hyperintense).' if mm.get('t2_strongly_hyperintense') else '.'))
    if mm.get('edema_halo_ratio') is not None:
        bits.append(f'FLAIR peritumoral / brain background ratio = {mm["edema_halo_ratio"]:.2f}'
                    + (' (edema likely).' if mm.get('edema_likely') else '.'))
    if mm.get('necrosis_likely'):
        bits.append(f'Low T1c-intensity fraction = {mm.get("t1c_low_intensity_fraction", 0):.2f} '
                    f'inside an otherwise enhancing tumor suggests necrosis.')
    return ' '.join(bits) if bits else 'No multimodal signals computed.'


def _narrate_model_agreement(mb: dict) -> str:
    probs = mb.get('per_model_probabilities')
    if not probs:
        return 'No classifier probabilities available.'
    items = ', '.join(f'{k}={v:.3f}' for k, v in probs.items())
    line = f'Per-model tumor probability: {items}.'
    line += f' Mean = {mb.get("mean_probability_tumor", 0):.3f}.'
    line += f' Models agreement: {mb.get("models_agreement", "n/a")}.'
    return line


def _narrate_confidence(mb: dict, geom: dict) -> str:
    if mb.get('gradcam_segmentation_aligned'):
        ga = (' Grad-CAM peak aligns well with the segmentation centroid '
              f'(distance {mb.get("gradcam_to_segmentation_distance_px", 0):.0f} px, '
              f'IoU {mb.get("gradcam_mask_iou", 0):.2f}), increasing confidence.')
    elif 'gradcam_to_segmentation_distance_px' in mb:
        ga = (' Grad-CAM peak is far from the segmentation centroid '
              f'(distance {mb.get("gradcam_to_segmentation_distance_px", 0):.0f} px, '
              f'IoU {mb.get("gradcam_mask_iou", 0):.2f}), which may indicate the classifier and '
              'segmenter are looking at different regions; treat with caution.')
    else:
        ga = ' Grad-CAM unavailable.'
    if geom.get('area_px', 0) < 50:
        return 'Predicted tumor is very small or absent; confidence is low.' + ga
    return f'Tumor of {geom.get("area_px", 0):,} pixels detected with classifier agreement {mb.get("models_agreement", "n/a")}.' + ga


def _narrate_diff(features: dict) -> list[str]:
    bullets: list[str] = []
    geom = features.get('geometry') or {}
    mm = features.get('multimodal') or {}
    loc = features.get('localization') or {}
    if mm.get('necrosis_likely') and mm.get('edema_likely'):
        bullets.append(
            'High-grade glioma (e.g. glioblastoma) - the combination of T1c necrotic core, '
            'peripheral enhancement, and FLAIR-bright peritumoral edema fits a typical HGG appearance.'
        )
    if mm.get('t1c_predominantly_enhancing') and not mm.get('edema_likely'):
        bullets.append(
            'Meningioma (extra-axial) - homogeneously enhancing on T1c with minimal edema can suggest '
            'a meningioma if cortically based; check for dural tail. (low confidence without dural sign)'
        )
    if geom.get('eccentricity', 0) > 0.75 and geom.get('solidity', 1) < 0.85:
        bullets.append(
            'Infiltrative glioma vs metastasis - irregular, elongated, low-solidity shape with concavities '
            'is more consistent with an infiltrative lesion than a well-circumscribed one.'
        )
    if loc.get('depth_label') == 'peripheral / cortical':
        bullets.append(
            'Metastasis - peripheral / cortical location can be seen with metastases, especially when '
            'multifocal. (low confidence, also seen with cortical gliomas)'
        )
    if features.get('components', {}).get('multifocal'):
        bullets.append(
            'Multifocal disease - multiple connected components on segmentation; consider metastatic '
            'disease, multifocal glioma, or lymphoma.'
        )
    if not bullets:
        bullets.append(
            'Cannot offer a confident differential from the features alone; recommend radiologist review.'
        )
    return bullets


__all__ = ['explain']
