import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
from src.models import get_model
from src.utils import load_metrics_json

MODEL_TYPES = ['cnn', 'transfer', 'vit']

# Search order for weights / metrics (best-trained snapshot first).
ARTIFACT_DIR_NAMES = ['real_eval_fixed', 'real_eval_current', 'artifacts']


def page_style():
    return """
    <style>
        .main { background: linear-gradient(135deg, #020617, #0f172a); color: #e2e8f0; }
        .title-block { padding: 1.4rem 1.5rem; border-radius: 24px; background: rgba(15, 23, 42, 0.95); box-shadow: 0 30px 60px rgba(0, 0, 0, 0.25); }
        .hero-title { font-size: 3rem; font-weight: 800; margin: 0; color: #f8fafc; }
        .hero-subtitle { color: #94a3b8; margin-top: 0.75rem; font-size: 1rem; line-height: 1.7; }
        .metric-card { padding: 1.2rem 1.3rem; border-radius: 20px; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.12); transition: transform 0.25s ease, box-shadow 0.25s ease; }
        .metric-card:hover { transform: translateY(-8px); box-shadow: 0 18px 40px rgba(15, 23, 42, 0.4); }
        .metric-title { font-size: 0.9rem; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8; margin-bottom: 0.45rem; }
        .metric-value { font-size: 2.1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.2rem; }
        .metric-detail { font-size: 0.88rem; color: #cbd5e1; }
        .prediction-card { padding: 1rem; border-radius: 18px; background: rgba(2, 12, 27, 0.92); border: 1px solid rgba(56, 189, 248, 0.2); }
        .table-zone { padding: 1rem; border-radius: 20px; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.12); }
        .upload-box { border: 2px dashed rgba(56, 189, 248, 0.45); border-radius: 18px; padding: 1.2rem; background: rgba(2, 12, 27, 0.9); }
        .upload-box:hover { background: rgba(2, 12, 27, 0.98); }
        .label-pill { display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px; background: #2563eb; color: #ffffff; font-size: 0.8rem; margin-right: 0.45rem; }
        .section-heading { margin-bottom: 0.75rem; font-weight: 700; color: #ffffff; }
        .footnote { color: #94a3b8; font-size: 0.9rem; margin-top: 1rem; }
        .centered-row { display: flex; gap: 1rem; flex-wrap: wrap; }
        .panel-title { font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 0.75rem; }
    </style>
    """


def html_header():
    return """
    <div class='title-block'>
      <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;'>
        <div style='max-width:70%;'>
          <div class='hero-title'>OncoVision</div>
          <div class='hero-subtitle'>A modern MRI brain tumor dashboard with compact model comparison, upload-driven inference, and performance metrics for CNN, Transfer, and Vision Transformer models.</div>
        </div>
        <div style='text-align:right;'>
          <span class='label-pill'>Oncology AI</span>
          <span class='label-pill'>Model comparison</span>
        </div>
      </div>
    </div>
    """


def _resolve_artifact_dirs(primary):
    """Return the ordered list of dirs we will probe for weights/metrics.

    The Streamlit sidebar lets the user override the primary directory, but if
    the user-provided one has no weights, we fall back to the shipped
    real_eval_* snapshots (the only ones that exist in a fresh clone).
    """
    primary = Path(primary)
    dirs = [primary]
    repo_root = Path(__file__).resolve().parent
    for name in ARTIFACT_DIR_NAMES:
        candidate = repo_root / name
        if candidate.exists() and candidate not in dirs:
            dirs.append(candidate)
    return dirs


def load_model_weights(model_type, artifacts_dir):
    """Find best_weights.weights.h5 (or any *.weights.h5) for the given model.

    Probes the user-provided artifacts_dir first, then falls back to
    real_eval_fixed/, real_eval_current/, and artifacts/ so the dashboard works
    out of the box with the snapshot weights that ship with this repo.
    """
    for base in _resolve_artifact_dirs(artifacts_dir):
        model_dir = base / model_type
        if not model_dir.exists():
            continue
        for candidate in (
            model_dir / 'best_weights.weights.h5',
            model_dir / 'best_weights.h5',
        ):
            if candidate.exists():
                return candidate
        for candidate in model_dir.glob('*.weights.h5'):
            return candidate
    return None


def load_model_metrics(model_type, artifacts_dir):
    """Find <model>_evaluation_metrics.json across the configured artifact dirs."""
    for base in _resolve_artifact_dirs(artifacts_dir):
        path = base / f'{model_type}_evaluation_metrics.json'
        if path.exists():
            return load_metrics_json(path)
    return None


def metric_summary(metrics):
    summary = {
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1_score': None,
        'roc_auc': None,
    }
    if not metrics:
        return summary
    summary['roc_auc'] = metrics.get('roc_auc')
    report = metrics.get('classification_report', {})
    if isinstance(report, dict):
        weighted = report.get('weighted avg', report.get('weighted_avg', {}))
        summary['accuracy'] = report.get('accuracy')
        summary['precision'] = weighted.get('precision')
        summary['recall'] = weighted.get('recall')
        summary['f1_score'] = weighted.get('f1-score', weighted.get('f1_score'))
    return summary


def available_models(artifacts_dir):
    return [m for m in MODEL_TYPES if load_model_weights(m, artifacts_dir) or load_model_metrics(m, artifacts_dir)]


def run_prediction(model_type, weight_path, image):
    model = get_model(model_type)
    model.load_weights(weight_path)
    x = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
    score = model.predict(np.expand_dims(x, axis=0), verbose=0)[0][0]
    return score


def render_metric_card(title, value, detail=''):
    return f"""
    <div class='metric-card'>
      <div class='metric-title'>{title}</div>
      <div class='metric-value'>{value}</div>
      <div class='metric-detail'>{detail}</div>
    </div>
    """


def render_comparison_table(table_data):
    if not table_data:
        return '<div style="color:#cbd5e1;">No model comparison metrics available yet.</div>'
    rows = ''.join(
        f"<tr style='border-bottom:1px solid rgba(148,163,184,0.18);'><td style='padding:0.9rem 0.7rem;'>{row['model'].upper()}</td><td style='padding:0.9rem 0.7rem;text-align:right;'>{row['Accuracy']:.3f}</td><td style='padding:0.9rem 0.7rem;text-align:right;'>{row['Precision']:.3f}</td><td style='padding:0.9rem 0.7rem;text-align:right;'>{row['Recall']:.3f}</td><td style='padding:0.9rem 0.7rem;text-align:right;'>{row['F1 Score']:.3f}</td><td style='padding:0.9rem 0.7rem;text-align:right;'>{row['ROC AUC']:.3f}</td></tr>"
        for row in table_data
    )
    return f"""
      <div class='table-zone'>
        <div class='panel-title'>Model comparison results</div>
        <table style='width:100%; border-collapse: collapse; color:#e2e8f0;'>
          <thead>
            <tr style='color:#94a3b8;'>
              <th style='text-align:left; padding: 0.9rem;'>Model</th>
              <th style='padding: 0.9rem; text-align:right;'>Accuracy</th>
              <th style='padding: 0.9rem; text-align:right;'>Precision</th>
              <th style='padding: 0.9rem; text-align:right;'>Recall</th>
              <th style='padding: 0.9rem; text-align:right;'>F1 Score</th>
              <th style='padding: 0.9rem; text-align:right;'>ROC AUC</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    """


def build_comparison_rows(artifacts_dir):
    rows = []
    for model_type in MODEL_TYPES:
        metrics = load_model_metrics(model_type, artifacts_dir)
        if metrics:
            summary = metric_summary(metrics)
            rows.append({
                'model': model_type,
                'Accuracy': summary['accuracy'] or 0.0,
                'Precision': summary['precision'] or 0.0,
                'Recall': summary['recall'] or 0.0,
                'F1 Score': summary['f1_score'] or 0.0,
                'ROC AUC': summary['roc_auc'] or 0.0,
            })
    return rows


def main():
    st.set_page_config(page_title='OncoVision', layout='wide')
    st.markdown(page_style(), unsafe_allow_html=True)
    components.html(html_header(), height=180)

    artifacts_dir = Path(st.sidebar.text_input('Artifacts directory', 'artifacts'))
    available = available_models(artifacts_dir)
    st.sidebar.markdown('---')
    st.sidebar.write('Available models:')
    for model_name in MODEL_TYPES:
        status = '✅' if model_name in available else '❌'
        st.sidebar.write(f'{status} {model_name.upper()}')
    st.sidebar.markdown('---')
    uploaded_file = st.sidebar.file_uploader('Upload MRI image', type=['jpg', 'jpeg', 'png'])

    metrics_rows = build_comparison_rows(artifacts_dir)
    best_model = max(metrics_rows, key=lambda item: item['Accuracy'] or 0) if metrics_rows else None

    left, right = st.columns([2, 1], gap='large')
    with left:
        st.markdown("<div class='section-heading'>Performance Overview</div>", unsafe_allow_html=True)
        metric_cards = []
        if best_model:
            metric_cards.append(render_metric_card('Top model', best_model['model'].upper(), 'Highest accuracy available'))
            metric_cards.append(render_metric_card('Accuracy', f"{best_model['Accuracy']:.3f}"))
            metric_cards.append(render_metric_card('Precision', f"{best_model['Precision']:.3f}"))
            metric_cards.append(render_metric_card('Recall', f"{best_model['Recall']:.3f}"))
            metric_cards.append(render_metric_card('F1 Score', f"{best_model['F1 Score']:.3f}"))
        else:
            metric_cards.append(render_metric_card('Ready to visualize', 'No models yet', 'Add metrics or weight files to ./artifacts'))
        st.markdown('<div class="centered-row">' + ''.join(metric_cards) + '</div>', unsafe_allow_html=True)

        st.markdown("<div class='section-heading'>Model Comparison</div>", unsafe_allow_html=True)
        st.components.v1.html(render_comparison_table(metrics_rows), height=320)

    with right:
        st.markdown("<div class='section-heading'>Image Prediction</div>", unsafe_allow_html=True)
        if uploaded_file is None:
            st.markdown("<div class='upload-box'><strong>Upload an MRI image</strong><br/>Drop a PNG or JPG scan to compare predictions across models.</div>", unsafe_allow_html=True)
        else:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded MRI image', use_column_width=True)
            if available:
                predictions = []
                for model_name in available:
                    weight_path = load_model_weights(model_name, artifacts_dir)
                    if weight_path:
                        score = run_prediction(model_name, weight_path, image)
                        predictions.append((model_name.upper(), score))
                if predictions:
                    cards = ''
                    for name, score in predictions:
                        label = 'TUMOR' if score >= 0.5 else 'NO TUMOR'
                        cards += f"<div class='prediction-card'><div class='metric-title'>{name}</div><div class='metric-value'>{score:.4f}</div><div class='metric-detail'>{label}</div></div>"
                    st.markdown('<div class="centered-row">' + cards + '</div>', unsafe_allow_html=True)
                else:
                    st.warning('No trained weights found to run predictions.')
            else:
                st.warning('No available models found. Add model artifacts under ./artifacts.')

    st.markdown("<div class='footnote'>OncoVision - compact and attractive MRI model comparison for end users.</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
