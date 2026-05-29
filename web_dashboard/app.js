/**
 * NeuroLens AI - Professional Dashboard Application
 */

class NeuroLensApp {
    constructor() {
        this.currentFile = null;
        this.currentResults = null;
        this.currentSegmentation = null;
        this.startTime = null;
        this.imageDataUrl = null;

        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadMetrics();
        this.setupNavigation();
    }
    
    bindEvents() {
        // File upload
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFile(e.target.files[0]);
            }
        });
        
        analyzeBtn.addEventListener('click', () => this.runAnalysis());
        
        // Navigation
        document.getElementById('newAnalysisBtn').addEventListener('click', () => {
            this.showSection('upload');
        });
        
        document.getElementById('exportBtn').addEventListener('click', () => {
            this.exportReport();
        });
        
        // Threshold slider
        const thresholdSlider = document.getElementById('thresholdSlider');
        const thresholdValue = document.getElementById('thresholdValue');
        
        thresholdSlider.addEventListener('input', (e) => {
            thresholdValue.textContent = (e.target.value / 100).toFixed(2);
        });
        thresholdSlider.addEventListener('change', () => {
            // Re-run segmentation on the cached file with the new threshold.
            if (this.currentFile) {
                this.runSegmentation();
            }
        });
        
        // Segmentation button
        document.getElementById('runSegmentationBtn').addEventListener('click', () => {
            this.runSegmentation();
        });

        // AI Explanation button
        const explainBtn = document.getElementById('runExplainBtn');
        if (explainBtn) {
            explainBtn.addEventListener('click', () => this.runExplanation());
        }
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleTabClick(e));
        });
        
        // Sidebar navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const tab = item.dataset.tab;
                
                this.showSection(tab);
                
                document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
                item.classList.add('active');
            });
        });
    }
    
    handleFile(file) {
        this.currentFile = file;
        
        // Update file info
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.imageDataUrl = e.target.result;
            const img = document.getElementById('previewImage');
            img.src = e.target.result;
            img.style.display = 'block';
            document.querySelector('.preview-placeholder').style.display = 'none';
            
            // Get image dimensions
            const tempImg = new Image();
            tempImg.onload = () => {
                document.getElementById('dimensions').textContent = `${tempImg.width} × ${tempImg.height}`;
            };
            tempImg.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Enable analyze button
        document.getElementById('analyzeBtn').disabled = false;
    }
    
    async runAnalysis() {
        if (!this.currentFile) return;

        const modelSelect = document.getElementById('modelSelect');
        const patientId = document.getElementById('patientId').value || `SCAN-${Date.now()}`;

        this.showLoading();
        this.startTime = Date.now();

        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        let progress = 0;
        const progressInterval = setInterval(() => {
            progress = Math.min(95, progress + Math.random() * 10 + 3);
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `Processing: ${Math.round(progress)}%`;
        }, 300);

        try {
            const selected = modelSelect.value || 'all';
            const modelsToCall = selected === 'all'
                ? ['cnn', 'transfer', 'vit']
                : [selected];

            // Kick everything off in parallel so a single click runs:
            //   - /metrics (cached per-model accuracy / AUC for the table)
            //   - one /predict per classifier (CNN / Transfer / ViT)
            //   - one /segment for the Attention U-Net mask + overlay
            // The user no longer has to click two buttons on two tabs.
            const thresholdInput = document.getElementById('thresholdSlider');
            const threshold = thresholdInput ? (parseInt(thresholdInput.value, 10) / 100) : 0.5;

            const metricsP = this.fetchMetricsByModel();
            const predictionPromises = modelsToCall.map(m =>
                this.callPredict(m, this.currentFile)
                    .then(result => ({ model: m, result, error: null }))
                    .catch(err => ({ model: m, result: null, error: err.message || String(err) }))
            );
            const segModalitySel = document.getElementById('segModelSelect');
            const segModality = segModalitySel ? segModalitySel.value : '';
            const segmentationP = this.callSegment(this.currentFile, threshold, segModality)
                .then(result => ({ result, error: null }))
                .catch(err => ({ result: null, error: err.message || String(err) }));

            const [metricsByModel, ...rest] = await Promise.all([metricsP, ...predictionPromises, segmentationP]);
            const segmentation = rest.pop();
            const predictionResults = rest;

            clearInterval(progressInterval);
            progressFill.style.width = '100%';
            progressText.textContent = 'Processing: 100%';

            this.currentResults = this.buildResultsFromBackend(patientId, predictionResults, metricsByModel);
            this.currentSegmentation = segmentation;

            setTimeout(() => {
                this.hideLoading();
                this.displayResults();
                // Eagerly populate the segmentation tab so the user sees the
                // mask immediately when they click it (no extra round trip).
                this.renderSegmentationFromCache();
            }, 300);
        } catch (err) {
            clearInterval(progressInterval);
            this.hideLoading();
            alert('Prediction failed: ' + (err.message || err));
            console.error(err);
        }
    }

    async callSegment(file, threshold = 0.5, modality = '') {
        const form = new FormData();
        form.append('image', file, file.name || 'upload.png');
        form.append('threshold', String(threshold));
        if (modality) form.append('modality', modality);
        const resp = await fetch('/segment', { method: 'POST', body: form });
        if (!resp.ok) {
            throw new Error(`/segment returned ${resp.status}`);
        }
        const payload = await resp.json();
        if (!payload || payload.success === false) {
            throw new Error((payload && payload.error) || '/segment failed');
        }
        return payload;
    }

    async callPredict(modelName, file) {
        const form = new FormData();
        form.append('model', modelName);
        form.append('image', file, file.name || 'upload.png');
        const resp = await fetch('/predict', { method: 'POST', body: form });
        if (!resp.ok) {
            throw new Error(`/predict ${modelName} returned ${resp.status}`);
        }
        const payload = await resp.json();
        if (!payload || payload.success === false) {
            throw new Error((payload && payload.error) || `/predict ${modelName} failed`);
        }
        return payload.result;
    }

    async fetchMetricsByModel() {
        try {
            const resp = await fetch('/metrics');
            if (!resp.ok) return {};
            return await resp.json();
        } catch (_) {
            return {};
        }
    }

    buildResultsFromBackend(patientId, predictionResults, metricsByModel) {
        const labelMap = {
            cnn: 'CNN (Fast)',
            transfer: 'Transfer Learning',
            vit: 'Vision Transformer',
        };

        const modelResults = predictionResults
            .filter(pr => pr.result)
            .map(pr => {
                const r = pr.result;
                const accuracy = metricsByModel?.[pr.model]?.metrics?.accuracy ?? null;
                const auc = metricsByModel?.[pr.model]?.metrics?.roc_auc ?? null;
                return {
                    model: pr.model,
                    modelLabel: labelMap[pr.model] || pr.model,
                    prediction: r.display_label || (r.label === 'tumor' ? 'Tumor' : 'No Tumor'),
                    confidence: r.confidence,
                    accuracy: accuracy,
                    auc: auc,
                    isPositive: r.label === 'tumor',
                    status: r.label === 'tumor' ? 'positive' : 'negative',
                    gradcam: r.gradcam || null,
                    image: r.image || null,
                    probability: r.probability,
                    weights: r.weights || null,
                };
            });

        // Errors get rendered as failed rows so the user can see what blew up.
        for (const pr of predictionResults.filter(pr => !pr.result)) {
            modelResults.push({
                model: pr.model,
                modelLabel: labelMap[pr.model] || pr.model,
                prediction: 'Error',
                confidence: 0,
                accuracy: null,
                auc: null,
                isPositive: false,
                status: 'negative',
                error: pr.error,
            });
        }

        // Best model = highest confidence among real (non-error) results.
        const realResults = modelResults.filter(r => !r.error);
        const bestModel = realResults.length
            ? realResults.reduce((best, cur) => (cur.confidence > best.confidence ? cur : best))
            : modelResults[0];

        const positiveVotes = realResults.filter(r => r.isPositive).length;
        const isPositive = positiveVotes >= Math.ceil(realResults.length / 2);
        const processingTime = ((Date.now() - this.startTime) / 1000).toFixed(1);

        return {
            patientId,
            timestamp: new Date().toLocaleString(),
            models: modelResults,
            bestModel,
            diagnosis: isPositive ? 'Tumor Detected' : 'No Tumor Detected',
            isPositive,
            confidence: bestModel ? bestModel.confidence : 0,
            processingTime,
            // Real uncertainty/robustness numbers are not computed by the
            // backend yet — the previous values were random JS placeholders.
            // We surface "N/A" rather than fabricate numbers.
            uncertainty: { epistemic: null, aleatoric: null },
            robustness: null,
        };
    }
    
    displayResults() {
        const results = this.currentResults;
        
        // Update subtitle
        document.getElementById('resultsSubtitle').textContent = 
            `Scan: ${results.patientId} · Analyzed at ${results.timestamp}`;
        
        // Update metrics
        document.getElementById('diagnosisValue').textContent = results.diagnosis;
        document.getElementById('diagnosisDetail').textContent = 'Requires clinical review';
        
        document.getElementById('confidenceValue').textContent = 
            `${(results.confidence * 100).toFixed(1)}%`;
        document.getElementById('confidenceFill').style.width = 
            `${results.confidence * 100}%`;
        
        document.getElementById('modelValue').textContent = 
            results.bestModel.modelLabel;
        
        document.getElementById('timeValue').textContent = 
            `${results.processingTime}s`;
        
        // Update comparison table with real metrics from /metrics. Accuracy and
        // AUC come from the persisted JSONs, not from the live prediction.
        const fmtPct = (v) => (v == null || Number.isNaN(v)) ? 'N/A' : `${(v * 100).toFixed(1)}%`;
        const tableBody = document.getElementById('comparisonTableBody');
        tableBody.innerHTML = results.models.map(model => `
            <tr class="${model === results.bestModel ? 'best' : ''}">
                <td><strong>${model.modelLabel}</strong></td>
                <td>${model.prediction}</td>
                <td>${fmtPct(model.confidence)}</td>
                <td>${fmtPct(model.accuracy)}</td>
                <td>${fmtPct(model.auc)}</td>
                <td>
                    <span class="status-badge ${model.status}">
                        ● ${model.status === 'positive' ? 'Positive' : 'Negative'}
                    </span>
                </td>
            </tr>
        `).join('');
        
        // Uncertainty / robustness numbers used to be JS placeholders. Until
        // a real estimator is wired up server-side we render N/A so the UI
        // does not lie about reliability.
        const epEl = document.getElementById('epistemicValue');
        const alEl = document.getElementById('aleatoricValue');
        epEl.textContent = results.uncertainty.epistemic == null ? 'N/A' : results.uncertainty.epistemic.toFixed(3);
        alEl.textContent = results.uncertainty.aleatoric == null ? 'N/A' : results.uncertainty.aleatoric.toFixed(3);
        const uncertaintyAvailable = results.uncertainty.epistemic != null && results.uncertainty.aleatoric != null;
        const totalUncertainty = uncertaintyAvailable
            ? (results.uncertainty.epistemic + results.uncertainty.aleatoric) / 2
            : 0;
        document.getElementById('uncertaintyFill').style.width = `${(uncertaintyAvailable ? totalUncertainty : 0) * 100}%`;
        document.getElementById('uncertaintyNote').textContent = uncertaintyAvailable
            ? (totalUncertainty < 0.1 ? 'Low uncertainty - High reliability'
                : totalUncertainty < 0.2 ? 'Moderate uncertainty - Review recommended'
                : 'High uncertainty - Clinical review required')
            : 'Uncertainty estimator not yet wired into the backend.';

        const robustnessPercent = results.robustness == null ? null : results.robustness * 100;
        document.getElementById('robustnessValue').textContent = robustnessPercent == null
            ? 'N/A'
            : `${robustnessPercent.toFixed(0)}%`;
        document.getElementById('robustnessGauge').style.background = robustnessPercent == null
            ? 'conic-gradient(var(--gray-200) 0deg, var(--gray-200) 360deg)'
            : `conic-gradient(var(--success) 0deg, var(--success) ${robustnessPercent * 3.6}deg, var(--gray-200) ${robustnessPercent * 3.6}deg)`;
        document.getElementById('robustnessNote').textContent = robustnessPercent == null
            ? 'Robustness analysis not yet wired into the backend.'
            : (robustnessPercent > 80 ? 'Excellent robustness'
                : robustnessPercent > 60 ? 'Good robustness'
                : 'Moderate robustness - Consider retraining');

        // Visualizations: the "Original" tab shows the uploaded image, the
        // Grad-CAM tabs show whatever the backend returned for the best model,
        // and the U-Net Mask / Overlay tabs show the segmentation result.
        if (this.imageDataUrl) {
            document.getElementById('vizImage').src = this.imageDataUrl;
        }
        this.setHeatmapFromBackend(results.bestModel);
        const seg = this.currentSegmentation;
        const maskImg = document.getElementById('maskImage');
        const segoverlayImg = document.getElementById('segoverlayImage');
        if (seg && seg.result && maskImg && segoverlayImg) {
            if (seg.result.mask) maskImg.src = seg.result.mask;
            if (seg.result.overlay) segoverlayImg.src = seg.result.overlay;
        } else if (maskImg && segoverlayImg) {
            maskImg.src = '';
            segoverlayImg.src = '';
        }
        
        // Show results section
        this.showSection('results');
    }
    
    setHeatmapFromBackend(bestModel) {
        // Real Grad-CAM data:url returned by /predict for cnn/transfer. For
        // ViT the backend returns null because the hybrid model has no single
        // canonical 'final conv' layer suitable for Grad-CAM. In that case we
        // fall back to displaying the input image with a small note.
        const heatmapImg = document.getElementById('heatmapImage');
        const overlayImg = document.getElementById('overlayImage');
        if (bestModel && bestModel.gradcam) {
            heatmapImg.src = bestModel.gradcam;
            overlayImg.src = bestModel.gradcam;
            return;
        }
        if (this.imageDataUrl) {
            heatmapImg.src = this.imageDataUrl;
            overlayImg.src = this.imageDataUrl;
        }
    }

    async runSegmentation() {
        if (!this.currentFile) {
            alert('Upload an MRI image first.');
            return;
        }
        const thresholdInput = document.getElementById('thresholdSlider');
        const thresholdValue = thresholdInput ? (parseInt(thresholdInput.value, 10) / 100) : 0.5;

        this.setSegmentationPanelLoading();
        try {
            const segModalitySel = document.getElementById('segModelSelect');
            const segModality = segModalitySel ? segModalitySel.value : '';
            const payload = await this.callSegment(this.currentFile, thresholdValue, segModality);
            this.currentSegmentation = { result: payload, error: null };
            this.renderSegmentationFromCache();
        } catch (err) {
            this.currentSegmentation = { result: null, error: err.message || String(err) };
            this.renderSegmentationFromCache();
            console.error(err);
        }
    }

    setSegmentationPanelLoading() {
        const segOriginal = document.getElementById('segOriginal');
        const segMask = document.getElementById('segMask');
        const segOverlay = document.getElementById('segOverlay');
        if (segOriginal && this.imageDataUrl) {
            segOriginal.innerHTML = `<img src="${this.imageDataUrl}" style="width:100%;height:100%;object-fit:contain;border-radius:12px;">`;
        }
        if (segMask) segMask.innerHTML = '<span style="opacity:0.6;">Running U-Net...</span>';
        if (segOverlay) segOverlay.innerHTML = '<span style="opacity:0.6;">Running U-Net...</span>';
        const dice = document.getElementById('diceScore');
        const iou = document.getElementById('iouScore');
        const area = document.getElementById('tumorArea');
        if (dice) dice.textContent = '...';
        if (iou) iou.textContent = '...';
        if (area) area.textContent = '...';
    }

    renderSegmentationFromCache() {
        const segOriginal = document.getElementById('segOriginal');
        const segMask = document.getElementById('segMask');
        const segOverlay = document.getElementById('segOverlay');
        const dice = document.getElementById('diceScore');
        const iou = document.getElementById('iouScore');
        const area = document.getElementById('tumorArea');
        if (!segMask) return;

        if (segOriginal && this.imageDataUrl) {
            segOriginal.innerHTML = `<img src="${this.imageDataUrl}" style="width:100%;height:100%;object-fit:contain;border-radius:12px;">`;
        }
        if (!this.currentSegmentation) {
            segMask.innerHTML = '<span style="opacity:0.6;">Upload an image and click "Run Analysis" to see the U-Net mask.</span>';
            segOverlay.innerHTML = '';
            return;
        }
        const seg = this.currentSegmentation;
        if (seg.error) {
            segMask.innerHTML = `<span style="color:#ef4444;">Error: ${seg.error}</span>`;
            segOverlay.innerHTML = '';
            if (dice) dice.textContent = '--';
            if (iou) iou.textContent = '--';
            if (area) area.textContent = '--';
            return;
        }
        const payload = seg.result || {};
        if (payload.mask) {
            segMask.innerHTML = `<img src="${payload.mask}" style="width:100%;height:100%;object-fit:contain;border-radius:12px;background:black;">`;
        }
        if (payload.overlay) {
            segOverlay.innerHTML = `<img src="${payload.overlay}" style="width:100%;height:100%;object-fit:contain;border-radius:12px;">`;
        }
        if (dice) dice.textContent = (payload.dice == null) ? 'N/A' : Number(payload.dice).toFixed(3);
        if (iou) iou.textContent = (payload.iou == null) ? 'N/A' : Number(payload.iou).toFixed(3);
        if (area) area.textContent = (payload.tumor_area_px == null) ? 'N/A' : `${payload.tumor_area_px} px`;

        // Cascade info: which checkpoint actually fired + why.
        const usedEl = document.getElementById('segUsedModel');
        const reasonEl = document.getElementById('segCascadeReason');
        const cascade = payload.cascade;
        if (usedEl) {
            const used = (cascade && cascade.used) || payload.source_dir || '--';
            // Make the label shorter and friendlier.
            const friendly = {
                'attention_unet_v3': 'v3 (multi-modal)',
                'attention_unet_v2': 'v2',
                'attention_unet_t1c': 'T1c specialist',
                'attention_unet_lgg': 'LGG',
                'attention_unet': 'baseline',
            };
            usedEl.textContent = friendly[used] || used;
        }
        if (reasonEl) {
            if (cascade && cascade.reason) {
                const reasonLabel = {
                    'v3_sufficient': 'v3 found enough tumor; no cascade',
                    'specialist_unavailable': 'T1c specialist checkpoint missing',
                    'explicit_modality_request': 'user picked this model',
                }[cascade.reason] || cascade.reason;
                reasonEl.textContent = reasonLabel;
            } else {
                reasonEl.textContent = '';
            }
        }
    }
    
    async callExplain(file, threshold, modality, backend) {
        const form = new FormData();
        form.append('image', file, file.name || 'upload.png');
        form.append('threshold', String(threshold));
        if (modality) form.append('modality', modality);
        if (backend) form.append('backend', backend);
        const resp = await fetch('/explain', { method: 'POST', body: form });
        if (!resp.ok) {
            throw new Error(`/explain returned ${resp.status}`);
        }
        const payload = await resp.json();
        if (!payload || payload.success === false) {
            throw new Error((payload && payload.error) || '/explain failed');
        }
        return payload;
    }

    async runExplanation() {
        if (!this.currentFile) {
            alert('Upload an MRI image first.');
            return;
        }
        const panel = document.getElementById('explainPanel');
        if (panel) panel.style.display = 'block';
        this.setExplanationLoading();

        const thresholdInput = document.getElementById('thresholdSlider');
        const threshold = thresholdInput ? (parseInt(thresholdInput.value, 10) / 100) : 0.5;
        const segModalitySel = document.getElementById('segModelSelect');
        const segModality = segModalitySel ? segModalitySel.value : '';
        const backendSel = document.getElementById('explainBackendSelect');
        const backend = backendSel ? backendSel.value : '';

        try {
            const payload = await this.callExplain(this.currentFile, threshold, segModality, backend);
            // Also update the segmentation viewers since /explain reran segmentation.
            if (payload.segmentation) {
                this.currentSegmentation = { result: payload.segmentation, error: null };
                this.renderSegmentationFromCache();
            }
            this.renderExplanation(payload);
        } catch (err) {
            console.error(err);
            this.renderExplanationError(err.message || String(err));
        }
    }

    setExplanationLoading() {
        const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
        set('explainBackend', 'running...');
        set('explainSafetyBadge', '');
        set('explainImpression', 'Calling LLM and extracting deterministic tumor features...');
        set('explainSummary', '...');
        set('explainAgreement', '...');
        set('explainConfidence', '...');
        set('explainConfBand', '--');
        set('explainConfScore', '--');
        set('explainGradeEvidence', '...');
        set('explainRecommendation', '...');
        set('explainDisclaimer', 'Not a medical diagnosis. Research / educational only.');
        const ids = ['explainFindings', 'explainDifferentialList', 'explainVisualObservations',
                     'explainVisualDisagreements', 'explainLlmPasses', 'explainQualityWarnings',
                     'explainRaw'];
        ids.forEach(id => { const el = document.getElementById(id); if (el) el.innerHTML = ''; });
        const fill = document.getElementById('explainConfFill');
        if (fill) fill.style.width = '0%';
    }

    renderExplanationError(message) {
        const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
        set('explainBackend', 'error');
        set('explainImpression', `Error: ${message}`);
        set('explainSummary', '--');
        set('explainAgreement', '--');
        set('explainConfidence', '--');
    }

    renderExplanation(payload) {
        const exp = (payload && payload.explanation) || {};
        const feats = (payload && payload.features) || {};
        const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text || '--'; };

        // --- Header (backend + safety badge) -------------------------------
        set('explainBackend', `${exp.backend || 'none'}${exp.model ? ` · ${exp.model}` : ''}`);
        const safety = exp.hallucination_safety || '';
        const safetyEl = document.getElementById('explainSafetyBadge');
        if (safetyEl) {
            const isZero = safety.toLowerCase().includes('guaranteed_zero');
            safetyEl.textContent = isZero ? 'Zero-Hallucination Mode' : 'Hallucination-Checked';
            safetyEl.title = safety;
            safetyEl.className = 'explain-safety-badge ' + (isZero ? 'safety-zero' : 'safety-checked');
        }

        // --- Impression + verified Summary --------------------------------
        set('explainImpression', exp.impression || exp.summary);
        set('explainSummary', exp.summary);
        set('explainDisclaimer', exp.disclaimer || 'Not a medical diagnosis. Research / educational only.');
        set('explainAgreement', exp.model_agreement_analysis);
        set('explainConfidence', exp.confidence_assessment);
        set('explainRecommendation', exp.recommendation);

        // --- Confidence band + score --------------------------------------
        const overall = feats.overall_confidence || {};
        const score = typeof overall.score_0_to_1 === 'number' ? overall.score_0_to_1 : null;
        const band = overall.band || '';
        const bandEl = document.getElementById('explainConfBand');
        if (bandEl) {
            bandEl.textContent = band || '--';
            bandEl.className = 'confidence-band conf-' + (band || 'unknown').replace(/[^a-z-]/gi, '');
        }
        const scoreEl = document.getElementById('explainConfScore');
        if (scoreEl) scoreEl.textContent = (score == null) ? '--' : `${(score * 100).toFixed(0)}%`;
        const fill = document.getElementById('explainConfFill');
        if (fill) fill.style.width = `${(score == null) ? 0 : score * 100}%`;

        // --- Structured findings (8 domains) ------------------------------
        const findings = document.getElementById('explainFindings');
        if (findings) {
            const fmap = exp.findings || {};
            const order = [
                ['geometry', 'Geometry'],
                ['localization', 'Localization'],
                ['intensity', 'Intensity'],
                ['texture', 'Texture'],
                ['multimodal', 'Multimodal'],
                ['morphology_margins', 'Morphology & Margins'],
                ['internal_architecture', 'Internal Architecture'],
                ['mass_effect', 'Mass Effect'],
            ];
            findings.innerHTML = order
                .filter(([k]) => fmap[k])
                .map(([k, label]) => `<dt>${label}</dt><dd>${this.escapeHtml(fmap[k])}</dd>`)
                .join('');
            if (!findings.innerHTML) {
                findings.innerHTML = '<dd style="opacity:0.6;">No structured findings returned.</dd>';
            }
        }

        // --- Grade evidence narrative -------------------------------------
        const gradeEl = document.getElementById('explainGradeEvidence');
        if (gradeEl) gradeEl.textContent = exp.grade_evidence_narrative || '--';

        // --- Differential with citations & origin tags --------------------
        const diff = document.getElementById('explainDifferentialList');
        if (diff) {
            const items = exp.differential_with_citations || [];
            if (items.length === 0) {
                diff.innerHTML = '<div style="opacity:0.6;">No differential hints returned.</div>';
            } else {
                diff.innerHTML = items.map(d => {
                    const origin = d.origin || 'rule-based';
                    const originLabel = origin === 'llm-citation-checked'
                        ? '<span class="origin-tag tag-llm">LLM · citation-checked</span>'
                        : '<span class="origin-tag tag-rule">Rule-based</span>';
                    const confTag = d.confidence
                        ? `<span class="conf-tag conf-${d.confidence.replace(/[^a-z-]/gi, '')}">${this.escapeHtml(d.confidence)}</span>`
                        : '';
                    const cites = (d.supported_by || []).map(c =>
                        `<code class="citation-chip">${this.escapeHtml(String(c))}</code>`
                    ).join(' ');
                    return `
                        <div class="differential-item">
                            <div class="differential-tags">${originLabel}${confTag}</div>
                            <div class="differential-statement">${this.escapeHtml(d.statement || '')}</div>
                            <div class="differential-citations">Supported by: ${cites || '<em>(no citations)</em>'}</div>
                        </div>`;
                }).join('');
            }
        }

        // --- Pattern C: visual observations -------------------------------
        const visualSection = document.getElementById('explainVisualSection');
        const visualList = document.getElementById('explainVisualObservations');
        const obs = exp.visual_observations || [];
        if (visualSection && visualList) {
            if (obs.length) {
                visualSection.style.display = '';
                visualList.innerHTML = obs.map(o => {
                    const region = this.escapeHtml(o.region || '?');
                    const claim = this.escapeHtml(o.claimed_property || '');
                    const text = this.escapeHtml(o.observation || '');
                    return `<li><strong>${region}</strong> — ${text} <span class="claim-prop">[${claim}]</span></li>`;
                }).join('');
            } else {
                visualSection.style.display = 'none';
            }
        }
        // Disagreements
        const disagreeSection = document.getElementById('explainDisagreementsSection');
        const disagreeList = document.getElementById('explainVisualDisagreements');
        const dis = exp.visual_disagreements || [];
        if (disagreeSection && disagreeList) {
            if (dis.length) {
                disagreeSection.style.display = '';
                disagreeList.innerHTML = dis.map(d => {
                    const text = this.escapeHtml(d.observation || '');
                    const conflicts = (d.conflicts_with || []).map(c => this.escapeHtml(c)).join('; ');
                    return `<li><strong>${text}</strong> <span class="claim-prop">conflicts with: ${conflicts}</span></li>`;
                }).join('');
            } else {
                disagreeSection.style.display = 'none';
            }
        }

        // --- LLM pass status (transparency) -------------------------------
        const passesEl = document.getElementById('explainLlmPasses');
        if (passesEl) {
            const passes = exp.llm_passes || {};
            const labels = {
                polish: 'Polish (Pattern A)',
                differential_expansion: 'Differential Expansion (Pattern B)',
                visual_observer: 'Visual Observer (Pattern C)',
            };
            const items = ['polish', 'differential_expansion', 'visual_observer']
                .filter(k => passes[k])
                .map(k => {
                    const p = passes[k];
                    const status = p.status || 'unknown';
                    const cssStatus = status.replace(/[^a-z_]/gi, '');
                    const model = p.model ? ` <span class="pass-model">${this.escapeHtml(p.model)}</span>` : '';
                    let detail = '';
                    if (status === 'error' || status === 'skipped_insufficient_ram') {
                        detail = `<div class="pass-detail pass-error">${this.escapeHtml(p.error || p.recovery_hint || '')}</div>`;
                    } else if (status === 'rejected') {
                        detail = `<div class="pass-detail pass-warn">Rejected: ${this.escapeHtml((p.warnings || []).join('; '))}</div>`;
                    } else if (status === 'ok' && k === 'differential_expansion') {
                        detail = `<div class="pass-detail">Accepted ${p.accepted_count || 0} · Rejected ${p.rejected_count || 0}</div>`;
                    } else if (status === 'ok' && k === 'visual_observer') {
                        detail = `<div class="pass-detail">${p.observation_count || 0} observations · ${p.disagreement_count || 0} disagreements</div>`;
                    } else if (status === 'skipped') {
                        detail = `<div class="pass-detail">${this.escapeHtml(p.reason || 'skipped')}</div>`;
                    }
                    return `
                        <div class="llm-pass-item pass-${cssStatus}">
                            <div class="pass-header">
                                <span class="pass-label">${labels[k]}</span>${model}
                            </div>
                            <div class="pass-status">${this.escapeHtml(status)}</div>
                            ${detail}
                        </div>`;
                }).join('');
            passesEl.innerHTML = items || '<div style="opacity:0.6;">No LLM passes run.</div>';
        }

        // --- Quality warnings ---------------------------------------------
        const qualSection = document.getElementById('explainQualitySection');
        const qualList = document.getElementById('explainQualityWarnings');
        const warnings = exp.quality_warnings || [];
        if (qualSection && qualList) {
            if (warnings.length) {
                qualSection.style.display = '';
                qualList.innerHTML = warnings.map(w => `<li>${this.escapeHtml(w)}</li>`).join('');
            } else {
                qualSection.style.display = 'none';
            }
        }

        // --- Raw features (collapsible) -----------------------------------
        const raw = document.getElementById('explainRaw');
        if (raw) {
            try { raw.textContent = JSON.stringify(feats, null, 2); }
            catch (_) { raw.textContent = String(feats); }
        }
    }

    escapeHtml(s) {
        if (s == null) return '';
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    handleTabClick(e) {
        const btn = e.target;
        const tabGroup = btn.parentElement;
        const tabType = btn.dataset.tab || btn.dataset.view;
        
        // Remove active from siblings
        tabGroup.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Handle view switching
        if (btn.dataset.view) {
            ['vizImage', 'heatmapImage', 'overlayImage', 'maskImage', 'segoverlayImage'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });
            // Map data-view names to img element ids.
            const idMap = {
                original: 'vizImage',
                heatmap: 'heatmapImage',
                overlay: 'overlayImage',
                mask: 'maskImage',
                segoverlay: 'segoverlayImage',
            };
            const targetId = idMap[tabType] || `${tabType}Image`;
            const target = document.getElementById(targetId);
            if (target) target.style.display = 'block';
        }
        
        // Handle comparison/details tab
        if (btn.dataset.tab === 'details') {
            this.showModelDetails();
        } else if (btn.dataset.tab === 'comparison') {
            this.showComparisonTable();
        }
    }
    
    showComparisonTable() {
        const content = document.getElementById('comparisonContent');
        if (this.currentResults) {
            content.innerHTML = `
                <div class="comparison-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Prediction</th>
                                <th>Confidence</th>
                                <th>Accuracy</th>
                                <th>ROC AUC</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${this.currentResults.models.map(model => {
                                const fmt = (v) => (v == null || Number.isNaN(v)) ? 'N/A' : `${(v * 100).toFixed(1)}%`;
                                return `
                                <tr class="${model === this.currentResults.bestModel ? 'best' : ''}">
                                    <td><strong>${model.modelLabel}</strong></td>
                                    <td>${model.prediction}</td>
                                    <td>${fmt(model.confidence)}</td>
                                    <td>${fmt(model.accuracy)}</td>
                                    <td>${fmt(model.auc)}</td>
                                    <td>
                                        <span class="status-badge ${model.status}">
                                            ● ${model.status === 'positive' ? 'Positive' : 'Negative'}
                                        </span>
                                    </td>
                                </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }
    }
    
    showModelDetails() {
        const content = document.getElementById('comparisonContent');
        if (this.currentResults) {
            content.innerHTML = `
                <div style="display: grid; gap: 20px;">
                    ${this.currentResults.models.map(model => {
                        const fmt = (v) => (v == null || Number.isNaN(v)) ? 'N/A' : `${(v * 100).toFixed(1)}%`;
                        return `
                        <div style="background: var(--gray-50); padding: 20px; border-radius: var(--radius-lg); border-left: 4px solid ${model === this.currentResults.bestModel ? 'var(--primary)' : 'var(--gray-300)'};">
                            <h4 style="margin-bottom: 12px; color: var(--gray-800);">${model.modelLabel}</h4>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                                <div>
                                    <div style="font-size: 12px; color: var(--gray-500); margin-bottom: 4px;">Confidence</div>
                                    <div style="font-size: 20px; font-weight: 700; color: var(--gray-800);">${fmt(model.confidence)}</div>
                                </div>
                                <div>
                                    <div style="font-size: 12px; color: var(--gray-500); margin-bottom: 4px;">Accuracy</div>
                                    <div style="font-size: 20px; font-weight: 700; color: var(--gray-800);">${fmt(model.accuracy)}</div>
                                </div>
                                <div>
                                    <div style="font-size: 12px; color: var(--gray-500); margin-bottom: 4px;">ROC AUC</div>
                                    <div style="font-size: 20px; font-weight: 700; color: var(--primary);">${fmt(model.auc)}</div>
                                </div>
                            </div>
                            <div style="margin-top: 12px;">
                                <div style="font-size: 12px; color: var(--gray-500); margin-bottom: 4px;">Prediction</div>
                                <span class="status-badge ${model.status}" style="font-size: 14px; padding: 6px 14px;">
                                    ● ${model.prediction}
                                </span>
                            </div>
                        </div>
                        `;
                    }).join('')}
                </div>
            `;
        }
    }
    
    showSection(section) {
        document.querySelectorAll('.content-section').forEach(s => {
            s.classList.remove('active');
            s.style.display = 'none';
        });
        
        const targetSection = document.getElementById(`${section}-section`);
        if (targetSection) {
            targetSection.classList.add('active');
            targetSection.style.display = 'block';
        }
    }
    
    showComingSoonMessage(title, description) {
        // Create a toast/notification element
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--gray-800);
            color: white;
            padding: 20px 24px;
            border-radius: var(--radius-lg);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            z-index: 10000;
            max-width: 400px;
            animation: slideIn 0.3s ease-out;
        `;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 24px; height: 24px; color: var(--primary);">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 16v-4"/>
                    <path d="M12 8h.01"/>
                </svg>
                <strong style="font-size: 16px;">${title}</strong>
            </div>
            <p style="font-size: 14px; opacity: 0.9; margin: 0;">${description}</p>
            <p style="font-size: 12px; opacity: 0.7; margin-top: 8px; margin-bottom: 0;">Coming Soon</p>
        `;
        
        // Add animation keyframes
        if (!document.getElementById('toast-styles')) {
            const style = document.createElement('style');
            style.id = 'toast-styles';
            style.textContent = `
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(toast);
        
        // Remove after 4 seconds
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
    
    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }
    
    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
    
    exportReport() {
        if (!this.currentResults) return;
        
        const report = {
            patientId: this.currentResults.patientId,
            timestamp: this.currentResults.timestamp,
            diagnosis: this.currentResults.diagnosis,
            confidence: this.currentResults.confidence,
            bestModel: this.currentResults.bestModel.modelLabel,
            modelResults: this.currentResults.models,
            uncertainty: this.currentResults.uncertainty,
            robustness: this.currentResults.robustness
        };
        
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${this.currentResults.patientId}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    async loadMetrics() {
        try {
            const response = await fetch('/metrics');
            if (response.ok) {
                const metrics = await response.json();
                console.log('Model metrics loaded:', metrics);
            }
        } catch (error) {
            console.log('Metrics not available (development mode)');
        }
    }
    
    setupNavigation() {
        // Handle recent scan clicks
        document.querySelectorAll('.recent-item').forEach(item => {
            item.addEventListener('click', () => {
                // In a real app, this would load the scan results
                alert('Loading scan results... (Feature coming soon)');
            });
        });
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    getModelLabel(model) {
        const labels = {
            'cnn': 'CNN (Fast)',
            'transfer': 'Transfer Learning',
            'vit': 'Vision Transformer',
            'attention_unet': 'Attention U-Net'
        };
        return labels[model] || model;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new NeuroLensApp();
});