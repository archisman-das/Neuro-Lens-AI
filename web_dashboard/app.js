const metricsSummary = document.getElementById('metricsSummary');
const metricsTableBody = document.querySelector('#metricsTable tbody');
const previewImage = document.getElementById('previewImage');
const emptyPreview = document.getElementById('emptyPreview');
const imageFrame = document.getElementById('imageFrame');
const imageInput = document.getElementById('imageInput');
const modelSelect = document.getElementById('modelSelect');
const predictButton = document.getElementById('predictButton');
const predictionResult = document.getElementById('predictionResult');
const uploadState = document.getElementById('uploadState');
const fileName = document.getElementById('fileName');
const statusCard = document.getElementById('statusCard');
const typeCard = document.getElementById('typeCard');
const confidenceCard = document.getElementById('confidenceCard');
const timeCard = document.getElementById('timeCard');
const thinkingState = document.getElementById('thinkingState');
const accuracyChart = document.getElementById('accuracyChart');
const zoomRange = document.getElementById('zoomRange');
const reportButton = document.getElementById('reportButton');
const reportState = document.getElementById('reportState');
const activityTimeline = document.getElementById('activityTimeline');
const runComparisonBtn = document.getElementById('runComparisonBtn');
const comparisonLoading = document.getElementById('comparisonLoading');
const comparisonResults = document.getElementById('comparisonResults');
const modelVisuals = document.getElementById('modelVisuals');

// Store last prediction results for comparative display
let lastPredictionResults = null;

const labels = {
  cnn: 'CNN',
  transfer: 'Transfer Learning',
  vit: 'Vision Transformer',
};

function formatNumber(value) {
  return typeof value === 'number' ? value.toFixed(3) : 'N/A';
}

function statusClass(entry) {
  if (entry.weights_found && entry.metrics_found) return 'ready';
  if (entry.weights_found || entry.metrics_found) return 'partial';
  return 'missing';
}

function statusText(entry) {
  if (entry.weights_found && entry.metrics_found) return 'Ready';
  if (entry.weights_found) return 'Weights only';
  if (entry.metrics_found) return 'Metrics only';
  return 'Not trained';
}

function addTimeline(title, detail) {
  const item = document.createElement('article');
  item.innerHTML = `<b>${title}</b><span>${detail}</span>`;
  activityTimeline.prepend(item);
}

function createMetricCard(entry) {
  const metric = entry.metrics || {};
  return `
    <article class="metric-card ${statusClass(entry)}">
      <h3>${entry.label}</h3>
      <small>${statusText(entry)}</small>
      <strong>${formatNumber(metric.accuracy)}</strong>
    </article>
  `;
}

function createAccuracyBar(entry) {
  const value = entry.metrics?.accuracy ?? 0;
  return `
    <div class="bar-row">
      <b>${entry.label}</b>
      <div class="bar-track"><span style="width:${Math.round(value * 100)}%"></span></div>
      <span>${formatNumber(value)}</span>
    </div>
  `;
}

function createResultCard(model, result) {
  const name = labels[model] || model;
  if (result.error) {
    return `
      <article class="result-card missing">
        <h3>${name}</h3>
        <p class="status-pill partial">Unavailable</p>
        <small>${result.hint || result.error}</small>
      </article>
    `;
  }

  const tumor = result.label === 'tumor';
  const probability = Math.max(0, Math.min(1, result.probability));
  return `
    <article class="result-card ${tumor ? 'alert' : 'clear'}">
      <h3>${name}</h3>
      <strong>${result.display_label}</strong>
      <div class="probability-row">
        <span>Tumor probability</span>
        <b>${formatNumber(result.probability)}</b>
      </div>
      <div class="confidence-bar" aria-hidden="true">
        <span style="width: ${Math.round(probability * 100)}%"></span>
      </div>
      <small>Confidence ${formatNumber(result.confidence)} · ${result.weights}</small>

  `;
}

async function fetchMetrics() {
  try {
    const response = await fetch('/metrics');
    const data = await response.json();
    renderMetrics(data);
  } catch (error) {
    metricsSummary.innerHTML = '<div class="empty-state">Unable to load evaluation metrics.</div>';
  }
}

function renderMetrics(data) {
  const entries = Object.values(data);
  metricsSummary.innerHTML = entries.map(createMetricCard).join('');
  accuracyChart.innerHTML = entries.map(createAccuracyBar).join('');
  metricsTableBody.innerHTML = entries.map((entry) => {
    const metric = entry.metrics || {};
    return `
      <tr>
        <td>${entry.label}</td>
        <td>${formatNumber(metric.accuracy)}</td>
        <td>${formatNumber(metric.precision)}</td>
        <td>${formatNumber(metric.recall)}</td>
        <td>${formatNumber(metric.f1_score)}</td>
        
        <td><span class="status-pill ${statusClass(entry)}">${statusText(entry)}</span></td>
      </tr>
    `;
  }).join('');
}

function previewUpload() {
  const file = imageInput.files[0];
  predictionResult.innerHTML = '<div class="empty-state">Run a scan to compare model outputs.</div>';
  statusCard.textContent = 'Pending';
  typeCard.textContent = 'Unknown';
  confidenceCard.textContent = '0.000';
  timeCard.textContent = '--';

  if (!file) {
    previewImage.removeAttribute('src');
    previewImage.alt = '';
    emptyPreview.hidden = false;
    fileName.textContent = 'No scan selected';
    uploadState.textContent = 'Select an MRI image to begin.';
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.alt = file.name;
    emptyPreview.hidden = true;
    fileName.textContent = file.name;
    uploadState.textContent = 'Scan loaded. Choose a model and run analysis.';
    imageFrame.classList.add('scanning');
    addTimeline('MRI uploaded', `${file.name} is ready for preprocessing.`);
  };
  reader.readAsDataURL(file);
}

function updateSummaryCards(result, elapsedMs) {
  let results = result;
  if (!Array.isArray(results)) {
    results = Object.values(result);
  }
  const valid = results.filter((item) => !item.error);
  if (!valid.length) return;

  const best = valid.reduce((winner, item) => item.confidence > winner.confidence ? item : winner, valid[0]);
  statusCard.textContent = best.label === 'tumor' ? 'Tumor' : 'Clear';
  typeCard.textContent = best.label === 'tumor' ? 'Tumor suspected' : 'No tumor';
  confidenceCard.textContent = formatNumber(best.confidence);
  timeCard.textContent = `${(elapsedMs / 1000).toFixed(1)}s`;
}

async function runPrediction() {
  const file = imageInput.files[0];
  if (!file) {
    uploadState.textContent = 'Please upload an MRI image first.';
    imageInput.focus();
    return;
  }

  const started = performance.now();
  const model = modelSelect.value;
  const formData = new FormData();
  formData.append('model', model);
  formData.append('image', file);

  predictButton.disabled = true;
  thinkingState.textContent = 'Analyzing';
  uploadState.textContent = 'Preprocessing scan and running model inference...';
  imageFrame.classList.add('scanning');
  predictionResult.innerHTML = '<div class="empty-state">AI thinking animation active. Running comparative analysis...</div>';

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();
    if (!data.success) {
      predictionResult.innerHTML = `<div class="empty-state">Prediction failed: ${data.error}</div>`;
      uploadState.textContent = 'Prediction failed.';
      return;
    }

    const elapsed = performance.now() - started;
    if (model === 'all') {
      predictionResult.innerHTML = Object.entries(data.result).map(([key, item]) => createResultCard(key, item)).join('');
      updateSummaryCards(Object.values(data.result), elapsed);
      if (modelVisuals) {
        const html = Object.entries(data.result).map(([key, item]) => {
          if (!item || !item.image) return '';
          return `<div style="text-align:center;min-width:200px;flex:1 1 220px;"><strong>${labels[key] || key}</strong><div style="margin-top:8px;"><img src="${item.image}" style="width:100%;border-radius:6px;display:block;"/></div>${item.gradcam?`<div style="margin-top:8px;"><img src="${item.gradcam}" style="width:100%;border-radius:6px;display:block;"/></div>`:''}</div>`;
        }).join('');
        modelVisuals.innerHTML = html;
      }
    } else {
      predictionResult.innerHTML = createResultCard(model, data.result);
      updateSummaryCards([data.result], elapsed);
      if (modelVisuals) {
        const item = data.result;
        modelVisuals.innerHTML = item && item.image ? `<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;"><div style="flex:1 1 320px;"><strong>Input</strong><img src="${item.image}" style="width:100%;border-radius:6px;margin-top:6px;display:block;"/></div>${item.gradcam?`<div style="flex:1 1 320px;"><strong>Grad-CAM</strong><img src="${item.gradcam}" style="width:100%;border-radius:6px;margin-top:6px;display:block;"/></div>`:''}</div>` : '';
      }
    }
    uploadState.textContent = 'Analysis complete. Report is ready.';
    addTimeline('Prediction complete', `Analysis finished in ${(elapsed / 1000).toFixed(1)} seconds.`);
  } catch (error) {
    predictionResult.innerHTML = `<div class="empty-state">Request error: ${error.message}</div>`;
    uploadState.textContent = 'Request error during prediction.';
  } finally {
    predictButton.disabled = false;
    thinkingState.textContent = 'Complete';
    imageFrame.classList.remove('scanning');
  }
}

function setView(mode) {
  document.querySelectorAll('.tool-button').forEach((button) => {
    button.classList.toggle('active', button.dataset.view === mode);
  });
}

document.querySelectorAll('.tool-button').forEach((button) => {
  button.addEventListener('click', () => setView(button.dataset.view));
});



zoomRange.addEventListener('input', () => {
  previewImage.style.transform = `scale(${zoomRange.value})`;
});

reportButton.addEventListener('click', () => {
  const patientId = (document.getElementById('patientId') || {}).value || 'patient';
  const now = new Date();
  const dateStr = now.toISOString();

  const reportHtml = `<!doctype html><html><head><meta charset="utf-8"><title>NeuroLens Report - ${patientId}</title>
    <style>body{font-family:Arial,Helvetica,sans-serif;padding:20px;color:#111} .section{margin-bottom:18px} h1{color:#0b1020}</style>
    </head><body>
    <h1>NeuroLens AI - Patient Report</h1>
    <div class="section"><strong>Patient ID:</strong> ${patientId}</div>
    <div class="section"><strong>Generated:</strong> ${dateStr}</div>
    <div class="section"><h2>Prediction Results</h2>${predictionResult.innerHTML}</div>
    <div class="section"><h2>Model Metrics</h2>${metricsSummary.innerHTML}</div>
    <div class="section"><em>Saved from NeuroLens AI local dashboard</em></div>
    </body></html>`;

  const blob = new Blob([reportHtml], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${patientId}_neuroLens_report_${now.toISOString().slice(0,19).replace(/[:T]/g,'-')}.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);

  reportState.textContent = `Report saved for ${patientId}`;
  addTimeline('Report generated', `Report saved for ${patientId} at ${dateStr}`);
});

imageInput.addEventListener('change', previewUpload);
predictButton.addEventListener('click', runPrediction);
// Comparative Analysis functionality - runs all three models on the same image
async function runComparativeAnalysis() {
  const file = imageInput.files[0];
  if (!file) {
    comparisonResults.innerHTML = '<div class="empty-state" style="color: var(--danger);">Please upload an MRI image first before running comparative analysis.</div>';
    return;
  }
  
  comparisonLoading.style.display = 'flex';
  comparisonResults.innerHTML = '';
  runComparisonBtn.disabled = true;
  
  const started = performance.now();
  const formData = new FormData();
  formData.append('model', 'all');
  formData.append('image', file);
  
  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();
    
    if (!data.success) {
      comparisonResults.innerHTML = `<div class="empty-state" style="color: var(--danger);">Prediction failed: ${data.error}</div>`;
      runComparisonBtn.disabled = false;
      comparisonLoading.style.display = 'none';
      return;
    }
    
    const elapsed = performance.now() - started;
    const results = data.result;
    lastPredictionResults = results;
    
    renderComparativeResults(results, elapsed);
    addTimeline('Comparative analysis complete', `All models analyzed ${file.name} in ${(elapsed / 1000).toFixed(1)} seconds.`);
    
  } catch (error) {
    comparisonResults.innerHTML = `<div class="empty-state" style="color: var(--danger);">Request error: ${error.message}</div>`;
  } finally {
    comparisonLoading.style.display = 'none';
    runComparisonBtn.disabled = false;
  }
}

function renderComparativeResults(results, elapsedMs) {
  const modelTypes = ['cnn', 'transfer', 'vit'];
  const modelNames = { cnn: 'CNN', transfer: 'Transfer Learning', vit: 'Vision Transformer' };
  const modelColors = { cnn: '#00d1ff', transfer: '#00e676', vit: '#ffb300' };
  
  let html = '';
  
  // Check model agreement
  const allLabels = modelTypes.map(m => results[m]?.label || 'unknown');
  const allSame = allLabels.every(l => l === allLabels[0]);
  const agreementCount = allSame ? 1 : 0;
  
  // Render each model's prediction
  modelTypes.forEach(model => {
    const pred = results[model];
    if (pred.error) {
      html += `
        <div class="comparison-card">
          <h4>${modelNames[model]}</h4>
          <div class="comparison-model-row">
            <span class="comparison-model-name">Status</span>
            <span style="color: var(--muted); font-size: 13px;">${pred.hint || 'Model unavailable'}</span>
            <span class="comparison-value" style="color: var(--danger);">--</span>
          </div>
        </div>
      `;
      return;
    }
    
    const isTumor = pred.label === 'tumor';
    const confidencePercent = Math.round(pred.confidence * 100);
    
    html += `
      <div class="comparison-card">
        <h4>${modelNames[model]}</h4>
        <div class="comparison-model-row">
          <span class="comparison-model-name">Diagnosis</span>
          <span style="color: ${isTumor ? 'var(--danger)' : 'var(--success)'}; font-weight: 700;">
            ${pred.display_label}
          </span>
          <span class="comparison-value" style="color: ${isTumor ? 'var(--danger)' : 'var(--success)'}">
            ${(pred.probability * 100).toFixed(1)}%
          </span>
        </div>
        <div class="comparison-model-row">
          <span class="comparison-model-name">Confidence</span>
          <div class="comparison-bar-track">
            <span style="width: ${confidencePercent}%; background: ${modelColors[model]}"></span>
          </div>
          <span class="comparison-value">${pred.confidence.toFixed(3)}</span>
        </div>
        <div class="comparison-model-row">
          <span class="comparison-model-name">Model Weights</span>
          <span style="color: var(--muted); font-size: 12px;">${pred.weights}</span>
        </div>
      </div>
    `;
  });
  
  // Summary section
  const validPredictions = modelTypes.filter(m => !results[m]?.error);
  const avgConfidence = {};
  validPredictions.forEach(model => {
    avgConfidence[model] = results[model].confidence.toFixed(3);
  });
  
  // Find highest confidence model
  let highestConfModel = null;
  let highestConf = 0;
  validPredictions.forEach(model => {
    if (results[model].confidence > highestConf) {
      highestConf = results[model].confidence;
      highestConfModel = model;
    }
  });
  
  html += `
    <div class="comparison-summary">
      <h4>Comparative Study Summary</h4>
      <p>
        ${validPredictions.length} of 3 models successfully analyzed the scan in ${(elapsedMs / 1000).toFixed(1)} seconds.
        ${allSame ? 'All models agree on the diagnosis.' : 'Models show disagreement in diagnosis - review recommended.'}
        ${highestConfModel ? `Highest confidence: ${modelNames[highestConfModel]} (${(highestConf * 100).toFixed(1)}%).` : ''}
      </p>
      <div class="comparison-legend">
        <div class="comparison-legend-item">
          <div class="comparison-legend-color" style="background: #00d1ff;"></div>
          <span>CNN</span>
        </div>
        <div class="comparison-legend-item">
          <div class="comparison-legend-color" style="background: #00e676;"></div>
          <span>Transfer Learning</span>
        </div>
        <div class="comparison-legend-item">
          <div class="comparison-legend-color" style="background: #ffb300;"></div>
          <span>Vision Transformer</span>
        </div>
      </div>
    </div>
  `;
  
  comparisonResults.innerHTML = html;
  // also populate the model visuals strip
  if (modelVisuals) {
    const visuals = Object.entries(results).map(([key, item]) => {
      if (!item || !item.image) return '';
      return `<div style="text-align:center;min-width:160px;margin-right:10px"><strong>${modelNames[key] || key}</strong><img src="${item.image}" style="width:160px;border-radius:6px;margin-top:6px;display:block;"/>${item.gradcam?`<img src="${item.gradcam}" style="width:160px;border-radius:6px;margin-top:6px;display:block;"/>`:''}</div>`;
    }).join('');
    modelVisuals.innerHTML = visuals;
  }
}

if (runComparisonBtn) {
  runComparisonBtn.addEventListener('click', runComparativeAnalysis);
}

window.addEventListener('load', fetchMetrics);
