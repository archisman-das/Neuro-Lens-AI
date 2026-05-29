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
        this.loadStatus();
        // Refresh the sidebar status every 30 s. Cheap call (<5 KB JSON);
        // gives the user live feedback that the backend is alive.
        if (!this._statusTimer) {
            this._statusTimer = setInterval(() => this.loadStatus(), 30_000);
        }
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

        const printBtn = document.getElementById('printBtn');
        if (printBtn) printBtn.addEventListener('click', () => this.printReport());
        
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

        // AI Explanation button on the Segmentation page.
        const explainBtn = document.getElementById('runExplainBtn');
        if (explainBtn) {
            explainBtn.addEventListener('click', () => this.runExplanation());
        }

        // AI Radiology Report button on the Results page - top-level surface
        // so the LLM explanation is one click away from the analysis the
        // user just ran.
        const generateBtn = document.getElementById('generateReportBtn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateReport());
        }

        // Batch upload: open multi-file picker -> sequential analysis.
        const batchBtn = document.getElementById('batchUploadBtn');
        const batchInput = document.getElementById('batchFileInput');
        if (batchBtn && batchInput) {
            batchBtn.addEventListener('click', () => batchInput.click());
            batchInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length) {
                    this.runBatchAnalysis(Array.from(e.target.files));
                }
                // reset so the same file can be re-selected
                e.target.value = '';
            });
        }
        const batchClearBtn = document.getElementById('batchClearBtn');
        if (batchClearBtn) batchClearBtn.addEventListener('click', () => this.clearBatch());
        const batchExportCsvBtn = document.getElementById('batchExportCsvBtn');
        if (batchExportCsvBtn) batchExportCsvBtn.addEventListener('click', () => this.exportBatchCsv());

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

            // Push to session-scoped Recent Scans sidebar.
            this.addRecentScan({
                id: patientId,
                isPositive: this.currentResults.isPositive,
                confidence: this.currentResults.confidence,
                timestamp: Date.now(),
            });

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
                    gradcam_heatmap: r.gradcam_heatmap || null,
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

        // ---- Classifier-ensemble reliability metrics ---------------------
        // Computed from the 3 per-model tumor probabilities:
        //   epistemic  = std of the 3 probabilities (inter-model disagreement)
        //   aleatoric  = binary entropy of the mean (data uncertainty near boundary)
        //   robustness = (max(mean, 1-mean) - 0.5) * 2 -> distance from boundary in [0,1]
        const probs = realResults
            .map(r => (typeof r.probability === 'number' ? r.probability : null))
            .filter(p => p != null);
        let epistemic = null, aleatoric = null, robustness = null, meanProb = null;
        if (probs.length >= 2) {
            const m = probs.reduce((a, b) => a + b, 0) / probs.length;
            const v = probs.reduce((a, b) => a + (b - m) * (b - m), 0) / probs.length;
            epistemic = Math.sqrt(v);
            // Binary entropy with safe log
            const eps = 1e-9;
            const safeM = Math.min(Math.max(m, eps), 1 - eps);
            aleatoric = -safeM * Math.log2(safeM) - (1 - safeM) * Math.log2(1 - safeM);
            // Distance from 0.5, scaled to [0,1]
            robustness = (Math.max(m, 1 - m) - 0.5) * 2;
            meanProb = m;
        }

        // ---- Classifier consensus verdict (matches the server-side rule) -
        // 'tumor' / 'no_tumor' / 'mixed' / null. Used to suppress the U-Net
        // false-positive mask when classifiers all agree it is not a tumor.
        let consensus = { verdict: null, mean: meanProb, band: null };
        if (probs.length >= 3 && meanProb != null) {
            const allAbove = probs.every(p => p >= 0.5);
            const allBelow = probs.every(p => p <= 0.5);
            if (meanProb >= 0.7 && allAbove) {
                consensus.verdict = 'tumor';
                consensus.band = meanProb >= 0.9 ? 'high' : 'moderate';
            } else if (meanProb <= 0.3 && allBelow) {
                consensus.verdict = 'no_tumor';
                consensus.band = meanProb <= 0.1 ? 'high' : 'moderate';
            } else {
                consensus.verdict = 'mixed';
                consensus.band = 'low';
            }
        }

        return {
            patientId,
            timestamp: new Date().toLocaleString(),
            models: modelResults,
            bestModel,
            diagnosis: isPositive ? 'Tumor Detected' : 'No Tumor Detected',
            isPositive,
            confidence: bestModel ? bestModel.confidence : 0,
            processingTime,
            consensus,
            uncertainty: { epistemic, aleatoric },
            robustness,
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
        
        // --- Uncertainty + Robustness (computed from the 3-classifier vote) ---
        const setT = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
        const fmt3 = (v) => (v == null || Number.isNaN(v)) ? 'N/A' : v.toFixed(3);
        const epEl = document.getElementById('epistemicValue');
        const alEl = document.getElementById('aleatoricValue');
        if (epEl) epEl.textContent = fmt3(results.uncertainty.epistemic);
        if (alEl) alEl.textContent = fmt3(results.uncertainty.aleatoric);
        const totalUnc = (results.uncertainty.epistemic == null || results.uncertainty.aleatoric == null)
            ? 0
            : Math.min(1, (results.uncertainty.epistemic + results.uncertainty.aleatoric) / 2);
        const uFill = document.getElementById('uncertaintyFill');
        if (uFill) uFill.style.width = `${totalUnc * 100}%`;
        const uNote = document.getElementById('uncertaintyNote');
        if (uNote) {
            if (results.uncertainty.epistemic == null) {
                uNote.textContent = 'Need >=2 classifier outputs to compute uncertainty.';
            } else if (totalUnc < 0.10) {
                uNote.textContent = 'Low total uncertainty - models confident, prediction near decision-boundary extreme.';
            } else if (totalUnc < 0.30) {
                uNote.textContent = 'Moderate uncertainty - clinical review recommended.';
            } else {
                uNote.textContent = 'High uncertainty - radiologist correlation required.';
            }
        }
        // Robustness (boundary distance) in [0,1] -> percent
        const robPct = results.robustness == null ? null : results.robustness * 100;
        const rValEl = document.getElementById('robustnessValue');
        if (rValEl) rValEl.textContent = robPct == null ? 'N/A' : `${robPct.toFixed(0)}%`;
        const rGauge = document.getElementById('robustnessGauge');
        if (rGauge) {
            rGauge.style.background = robPct == null
                ? 'conic-gradient(var(--gray-200) 0deg, var(--gray-200) 360deg)'
                : `conic-gradient(var(--success) 0deg, var(--success) ${robPct * 3.6}deg, var(--gray-200) ${robPct * 3.6}deg)`;
        }
        const rNote = document.getElementById('robustnessNote');
        if (rNote) {
            if (robPct == null) {
                rNote.textContent = 'Need >=2 classifier outputs to compute robustness.';
            } else if (robPct >= 90) {
                rNote.textContent = 'Excellent robustness - prediction far from decision boundary.';
            } else if (robPct >= 60) {
                rNote.textContent = 'Good robustness.';
            } else {
                rNote.textContent = 'Moderate robustness - prediction is close to the decision boundary.';
            }
        }

        // --- Inference telemetry + cascade decision ----------------------
        const segResult = this.currentSegmentation && this.currentSegmentation.result;
        const anyRuntime = (results.models.find(m => m.runtime) || {}).runtime
            || (segResult && segResult.runtime) || '--';
        setT('telemRuntime', anyRuntime);
        setT('telemTotal', `${results.processingTime}s`);
        if (segResult) {
            const cascade = segResult.cascade || {};
            setT('telemSegModel', cascade.used || segResult.source_dir || '--');
            setT('telemSegReason', cascade.reason || 'n/a');
            setT('telemSegArea', (segResult.tumor_area_px != null) ? `${segResult.tumor_area_px} px` : '--');
            setT('telemSegMeanProb', (segResult.mean_prob_in_mask != null) ? segResult.mean_prob_in_mask.toFixed(3) : '--');
        } else {
            ['telemSegModel', 'telemSegReason', 'telemSegArea', 'telemSegMeanProb']
                .forEach(id => setT(id, '--'));
        }

        // --- Visualizations ----------------------------------------------
        if (this.imageDataUrl) document.getElementById('vizImage').src = this.imageDataUrl;
        this.setHeatmapFromBackend(results.bestModel);

        // Apply the classifier-verdict gate. When all 3 classifiers agree NO
        // tumor with at least 'moderate' confidence, the U-Net mask is
        // treated as a probable false positive: the overlay tab shows the
        // ORIGINAL image with a warning badge, instead of the green-stained
        // overlay. The raw mask tab still shows the model's output for
        // transparency.
        const maskImg = document.getElementById('maskImage');
        const segoverlayImg = document.getElementById('segoverlayImage');
        const verdict = results.consensus && results.consensus.verdict;
        const verdictBand = results.consensus && results.consensus.band;
        const suppress = (verdict === 'no_tumor' && (verdictBand === 'high' || verdictBand === 'moderate'));
        this._maskSuppressed = suppress;
        this._maskSuppressedReason = suppress
            ? `Suppressed: classifier consensus is no-tumor (mean p=${results.consensus.mean.toFixed(3)}, ${verdictBand} confidence). U-Net masks on no-tumor scans are probable false positives - the U-Net was not trained on healthy brains.`
            : null;
        if (segResult && maskImg && segoverlayImg) {
            if (segResult.mask) maskImg.src = segResult.mask;
            // When suppressed: paint the original image instead of the green
            // overlay so the user does not get a false-positive visual signal.
            if (suppress && this.imageDataUrl) {
                segoverlayImg.src = this.imageDataUrl;
            } else if (segResult.overlay) {
                segoverlayImg.src = segResult.overlay;
            }
        } else if (maskImg && segoverlayImg) {
            maskImg.src = '';
            segoverlayImg.src = '';
        }
        
        // Show results section
        this.showSection('results');
    }
    
    setHeatmapFromBackend(bestModel) {
        // Real Grad-CAM data URL returned by /predict for cnn/transfer. The
        // hybrid ViT and the Spaces ONNX deploy both return null (no autograd
        // graph available). When null we show a true "unavailable" placeholder
        // instead of repeating the raw MRI, which previously was confusing.
        const heatmapImg = document.getElementById('heatmapImage');
        const overlayImg = document.getElementById('overlayImage');
        const placeholder = document.getElementById('vizPlaceholder');
        if (bestModel && bestModel.gradcam) {
            // Distinct images per tab. gradcam_heatmap is the pure colormap
            // (no MRI blended in) - shown on the "Grad-CAM" tab. gradcam is
            // the heatmap-blended-with-MRI - shown on the "Grad-CAM Overlay"
            // tab. Falling back to the overlay if the backend didn't split
            // (e.g. legacy TF .h5 path).
            heatmapImg.src = bestModel.gradcam_heatmap || bestModel.gradcam;
            overlayImg.src = bestModel.gradcam;
            heatmapImg.dataset.available = 'true';
            overlayImg.dataset.available = 'true';
        } else {
            // Clear the src and store an availability flag the tab-click
            // handler reads to swap in the placeholder.
            heatmapImg.src = '';
            overlayImg.src = '';
            heatmapImg.dataset.available = 'false';
            overlayImg.dataset.available = 'false';
            this._gradcamUnavailableReason = (bestModel && bestModel.runtime === 'onnx')
                ? 'Grad-CAM requires the PyTorch autograd graph and is not available in the ONNX-only deploy (this Space). Run the local dashboard with .pt weights to view Grad-CAM overlays.'
                : 'Grad-CAM unavailable for this model.';
            if (placeholder) {
                placeholder.textContent = this._gradcamUnavailableReason;
            }
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

    /**
     * Generate Report flow on the Results page. Calls /explain (which runs
     * the cascade segmentation + 3 classifiers + feature extraction + the
     * 3-pattern LLM pipeline), then renders the full explanation panel
     * inline inside #reportContent.
     */
    /**
     * Batch upload: process N files sequentially through the same /predict
     * + /segment pipeline used by Run Analysis, then render a comparison
     * table on the Upload section. Each row is clickable to deep-link into
     * the full Results view for that file. The selected backend / threshold
     * / model from the Upload form are honored for the whole batch.
     */
    async runBatchAnalysis(files) {
        if (!this._batchResults) this._batchResults = [];
        const panel = document.getElementById('batchPanel');
        const progressWrap = document.getElementById('batchProgressWrap');
        const progressFill = document.getElementById('batchProgressFill');
        const progressText = document.getElementById('batchProgressText');
        const tbody = document.getElementById('batchTableBody');
        const subtitle = document.getElementById('batchSubtitle');
        if (panel) panel.style.display = 'block';
        if (progressWrap) progressWrap.style.display = 'block';
        if (subtitle) subtitle.textContent = `${files.length} file${files.length === 1 ? '' : 's'} queued ...`;

        // Read upload form choices once so the whole batch uses the same setup.
        const modelSelect = document.getElementById('modelSelect');
        const modelChoice = modelSelect ? (modelSelect.value || 'all') : 'all';
        const segModalitySel = document.getElementById('segModelSelect');
        const segModality = segModalitySel ? segModalitySel.value : '';
        const thresholdInput = document.getElementById('thresholdSlider');
        const threshold = thresholdInput ? (parseInt(thresholdInput.value, 10) / 100) : 0.5;

        // Sequential processing keeps the small server stable. Cheap rows
        // (CPU-bound /predict on CNN classifier alone) finish in ~50 ms; a
        // full /predict 'all' + /segment is ~1-1.5 s. With N=8 the batch
        // completes in ~10 s.
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (progressText) progressText.textContent = `Processing ${i + 1} / ${files.length} - ${file.name}`;
            if (progressFill) progressFill.style.width = `${((i) / files.length) * 100}%`;
            const tStart = performance.now();
            try {
                const modelsToCall = modelChoice === 'all' ? ['cnn', 'transfer', 'vit'] : [modelChoice];
                const predictions = await Promise.all(modelsToCall.map(m =>
                    this.callPredict(m, file)
                        .then(result => ({ model: m, result, error: null }))
                        .catch(err => ({ model: m, result: null, error: err.message || String(err) }))
                ));
                const seg = await this.callSegment(file, threshold, segModality)
                    .then(result => ({ result, error: null }))
                    .catch(err => ({ result: null, error: err.message || String(err) }));

                const probs = predictions
                    .map(p => (p.result && typeof p.result.probability === 'number') ? p.result.probability : null)
                    .filter(p => p != null);
                const mean = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null;
                const std = (probs.length >= 2 && mean != null)
                    ? Math.sqrt(probs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / probs.length) : null;
                const safeM = mean == null ? null : Math.min(Math.max(mean, 1e-9), 1 - 1e-9);
                const entropy = safeM == null ? null
                    : -safeM * Math.log2(safeM) - (1 - safeM) * Math.log2(1 - safeM);

                let verdict = 'mixed', band = 'low';
                if (probs.length >= 3 && mean != null) {
                    const allAbove = probs.every(p => p >= 0.5);
                    const allBelow = probs.every(p => p <= 0.5);
                    if (mean >= 0.7 && allAbove) { verdict = 'tumor'; band = mean >= 0.9 ? 'high' : 'moderate'; }
                    else if (mean <= 0.3 && allBelow) { verdict = 'no_tumor'; band = mean <= 0.1 ? 'high' : 'moderate'; }
                }

                const elapsed = (performance.now() - tStart) / 1000;
                const best = predictions
                    .filter(p => p.result)
                    .reduce((acc, p) => (!acc || (p.result.confidence > acc.result.confidence) ? p : acc), null);
                const scanId = `BATCH-${Date.now()}-${i}`;
                // Read the file into a data URL once so the Results-page
                // preview can show the original MRI when the user drills in.
                let imageDataUrl = null;
                try {
                    imageDataUrl = await new Promise((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onload = () => resolve(reader.result);
                        reader.onerror = () => reject(reader.error);
                        reader.readAsDataURL(file);
                    });
                } catch (_) { /* ignore - preview just won't show */ }
                const entry = {
                    id: scanId, filename: file.name,
                    // Retain the File object so downstream actions on the
                    // Results page (Generate Report, Print, re-segment with
                    // a different threshold) can re-POST the bytes to the
                    // server. Without this, generateReport() bails because
                    // this.currentFile was never set.
                    file: file,
                    imageDataUrl,
                    predictions, segmentation: seg,
                    mean, std, entropy, verdict, band,
                    bestModel: best ? best.model : '--',
                    elapsedSeconds: elapsed.toFixed(2),
                    timestamp: Date.now(),
                };
                this._batchResults.push(entry);
                this.renderBatchRow(tbody, entry);
                // Also push to Recent Scans sidebar so it's discoverable.
                this.addRecentScan({
                    id: scanId,
                    isPositive: verdict === 'tumor',
                    confidence: mean || 0,
                    timestamp: Date.now(),
                });
            } catch (err) {
                console.error('Batch entry failed:', err);
            }
            if (progressFill) progressFill.style.width = `${((i + 1) / files.length) * 100}%`;
        }
        if (progressText) progressText.textContent = `Done. ${this._batchResults.length} total in batch.`;
        if (subtitle) {
            const tumorCount = this._batchResults.filter(e => e.verdict === 'tumor').length;
            const noTumorCount = this._batchResults.filter(e => e.verdict === 'no_tumor').length;
            const mixedCount = this._batchResults.length - tumorCount - noTumorCount;
            subtitle.textContent = `${this._batchResults.length} scans: ${tumorCount} tumor, ${noTumorCount} no-tumor, ${mixedCount} ambiguous.`;
        }
    }

    renderBatchRow(tbody, e) {
        if (!tbody) return;
        const idx = this._batchResults.length;
        const row = document.createElement('tr');
        row.dataset.batchId = e.id;
        const verdictBadge = `<span class="batch-verdict ${e.verdict}">${this.escapeHtml(e.verdict)}</span>`;
        const meanStr = e.mean == null ? '--' : e.mean.toFixed(3);
        const stdStr = e.std == null ? '--' : e.std.toFixed(3);
        const entStr = e.entropy == null ? '--' : e.entropy.toFixed(3);
        row.innerHTML = `
            <td>${idx}</td>
            <td title="${this.escapeHtml(e.filename)}">${this.escapeHtml(e.filename.length > 36 ? e.filename.slice(0, 33) + '...' : e.filename)}</td>
            <td><span class="batch-diag ${e.verdict === 'tumor' ? 'positive' : (e.verdict === 'no_tumor' ? 'negative' : 'mixed')}">${e.verdict === 'tumor' ? 'Tumor' : (e.verdict === 'no_tumor' ? 'No Tumor' : 'Mixed')}</span></td>
            <td>${this.escapeHtml(this.getModelLabel(e.bestModel))}</td>
            <td>${meanStr}</td>
            <td>${stdStr}</td>
            <td>${entStr}</td>
            <td>${verdictBadge} <span class="batch-band">${this.escapeHtml(e.band || '')}</span></td>
            <td>${e.elapsedSeconds}s</td>
            <td><button class="btn btn-small btn-outline" data-batch-view="${e.id}">View</button></td>
        `;
        const viewBtn = row.querySelector('[data-batch-view]');
        if (viewBtn) viewBtn.addEventListener('click', () => this.viewBatchEntry(e.id));
        tbody.appendChild(row);
    }

    viewBatchEntry(id) {
        const entry = (this._batchResults || []).find(e => e.id === id);
        if (!entry) return;
        // Rebuild the displayResults-compatible payload from the batch entry.
        const meanProb = entry.mean;
        const labelMap = { cnn: 'CNN (Fast)', transfer: 'Transfer Learning', vit: 'Vision Transformer' };
        const modelResults = entry.predictions.filter(p => p.result).map(p => ({
            model: p.model,
            modelLabel: labelMap[p.model] || p.model,
            prediction: p.result.display_label || (p.result.label === 'tumor' ? 'Tumor' : 'No Tumor'),
            confidence: p.result.confidence,
            accuracy: null, auc: null,
            isPositive: p.result.label === 'tumor',
            status: p.result.label === 'tumor' ? 'positive' : 'negative',
            gradcam: p.result.gradcam || null,
            gradcam_heatmap: p.result.gradcam_heatmap || null,
            image: p.result.image || null,
            probability: p.result.probability,
            runtime: p.result.runtime || null,
        }));
        const bestModel = modelResults.length
            ? modelResults.reduce((a, b) => (b.confidence > a.confidence ? b : a)) : null;
        const robustness = meanProb == null ? null : (Math.max(meanProb, 1 - meanProb) - 0.5) * 2;
        this.currentResults = {
            patientId: entry.id,
            timestamp: new Date(entry.timestamp).toLocaleString(),
            models: modelResults,
            bestModel,
            diagnosis: entry.verdict === 'tumor' ? 'Tumor Detected' : 'No Tumor Detected',
            isPositive: entry.verdict === 'tumor',
            confidence: bestModel ? bestModel.confidence : 0,
            processingTime: entry.elapsedSeconds,
            consensus: { verdict: entry.verdict, mean: meanProb, band: entry.band },
            uncertainty: { epistemic: entry.std, aleatoric: entry.entropy },
            robustness,
        };
        this.currentSegmentation = entry.segmentation || { result: null, error: null };
        // Restore the File object + data URL we captured at batch time so
        // downstream actions on the Results page (Generate Report, Print,
        // re-segment) can re-POST the bytes to /explain / /segment without
        // the user having to re-upload.
        this.currentFile = entry.file || null;
        this.imageDataUrl = entry.imageDataUrl
            || entry.predictions.find(p => p.result && p.result.image)?.result?.image
            || null;
        this.currentExplanation = null;  // not generated yet for this batch entry
        this.displayResults();
    }

    clearBatch() {
        this._batchResults = [];
        const panel = document.getElementById('batchPanel');
        if (panel) panel.style.display = 'none';
        const tbody = document.getElementById('batchTableBody');
        if (tbody) tbody.innerHTML = '';
    }

    exportBatchCsv() {
        const rows = this._batchResults || [];
        if (!rows.length) {
            this.showToast('Nothing to export', 'Run a batch upload first.', 'error');
            return;
        }
        const header = ['index', 'filename', 'verdict', 'band', 'mean_probability',
                         'std_probability', 'entropy', 'best_model',
                         'cnn_p', 'transfer_p', 'vit_p', 'elapsed_seconds', 'timestamp'];
        const lines = [header.join(',')];
        rows.forEach((e, i) => {
            const get = (m) => {
                const p = e.predictions.find(x => x.model === m);
                return (p && p.result && typeof p.result.probability === 'number')
                    ? p.result.probability.toFixed(4) : '';
            };
            const cells = [
                i + 1,
                `"${(e.filename || '').replace(/"/g, '""')}"`,
                e.verdict || '',
                e.band || '',
                e.mean == null ? '' : e.mean.toFixed(4),
                e.std == null ? '' : e.std.toFixed(4),
                e.entropy == null ? '' : e.entropy.toFixed(4),
                e.bestModel || '',
                get('cnn'), get('transfer'), get('vit'),
                e.elapsedSeconds || '',
                new Date(e.timestamp).toISOString(),
            ];
            lines.push(cells.join(','));
        });
        const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `neurolens_batch_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('CSV exported', `${rows.length} row${rows.length === 1 ? '' : 's'} saved.`, 'success');
    }

    async generateReport() {
        if (!this.currentFile) {
            this.showToast('Upload an image first', 'Run Analysis on an MRI before requesting the report.', 'error');
            return;
        }
        const placeholder = document.getElementById('reportPlaceholder');
        const content = document.getElementById('reportContent');
        const btn = document.getElementById('generateReportBtn');
        if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }

        // Build the rich panel skeleton inside reportContent. We literally
        // duplicate the explain panel markup so renderExplanation can target
        // the same element IDs as on the Segmentation tab.
        if (content) {
            content.style.display = 'block';
            content.innerHTML = this._explainPanelMarkup();
        }
        if (placeholder) placeholder.style.display = 'none';

        const thresholdInput = document.getElementById('thresholdSlider');
        const threshold = thresholdInput ? (parseInt(thresholdInput.value, 10) / 100) : 0.5;
        const backendSel = document.getElementById('reportBackendSelect');
        const backend = backendSel ? backendSel.value : '';

        try {
            const payload = await this.callExplain(this.currentFile, threshold, '', backend);
            if (payload.segmentation) {
                this.currentSegmentation = { result: payload.segmentation, error: null };
                this.renderSegmentationFromCache();
            }
            this.currentExplanation = payload.explanation || null;
            this.renderExplanation(payload);
            this.showToast('Report ready', `${(payload.explanation?.backend || 'deterministic')} backend completed.`, 'success');
        } catch (err) {
            console.error(err);
            this.renderExplanationError(err.message || String(err));
            this.showToast('Report failed', err.message || String(err), 'error');
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = 'Generate Report'; }
        }
    }

    /** Returns the same DOM IDs as #explainPanel so renderExplanation can target
     * them inside the Results-page report block. */
    _explainPanelMarkup() {
        return `
            <div class="explain-header">
                <h3>Layered Pipeline Output</h3>
                <div class="explain-header-meta">
                    <span class="explain-backend" id="explainBackend">--</span>
                    <span class="explain-safety-badge" id="explainSafetyBadge">--</span>
                </div>
            </div>
            <div class="explain-body">
                <div class="explain-section explain-impression">
                    <h4>Impression</h4>
                    <p id="explainImpression">--</p>
                </div>
                <div class="explain-section explain-confidence-card">
                    <h4>Overall Confidence</h4>
                    <div class="confidence-row">
                        <div class="confidence-band" id="explainConfBand">--</div>
                        <div class="confidence-score">
                            <div class="confidence-score-value" id="explainConfScore">--</div>
                            <div class="confidence-score-bar"><div class="confidence-score-fill" id="explainConfFill"></div></div>
                        </div>
                    </div>
                    <p id="explainConfidence" class="explain-confidence-detail">--</p>
                </div>
                <div class="explain-section explain-negative-explanation" id="explainNegativeSection" style="display:none;">
                    <h4>Why the Classifiers Ruled This Out</h4>
                    <pre id="explainNegativeExplanation" class="explain-grade"></pre>
                </div>
                <div class="explain-section explain-vision-negative" id="explainVisionNegativeSection" style="display:none;">
                    <h4>Vision LLM Reasoning (Pattern D - negative-case)</h4>
                    <p class="explain-subtle">The vision model was shown the original MRI and asked to describe what visible features support the no-tumor verdict. Output is validated to ensure it does not contradict the verdict.</p>
                    <pre id="explainVisionNegativeText" class="explain-grade"></pre>
                </div>
                <div class="explain-section" id="explainFindingsSection">
                    <h4>Structured Findings</h4>
                    <dl class="explain-findings" id="explainFindings"></dl>
                </div>
                <div class="explain-section" id="explainGradeSection">
                    <h4>Grade-Evidence Score</h4>
                    <pre id="explainGradeEvidence" class="explain-grade">--</pre>
                </div>
                <details class="explain-raw" id="explainFpRegionSection" style="display:none;">
                    <summary>False-positive region analysis (debug)</summary>
                    <dl class="explain-findings" id="explainFpRegionFindings"></dl>
                    <pre id="explainFpRegionGrade" class="explain-grade"></pre>
                </details>
                <div class="explain-section">
                    <h4>Differential Diagnosis (citation-checked)</h4>
                    <div id="explainDifferentialList" class="differential-list"></div>
                </div>
                <div class="explain-section" id="explainVisualSection" style="display:none;">
                    <h4>Visual Observations (LLM co-observer)</h4>
                    <ul id="explainVisualObservations"></ul>
                </div>
                <div class="explain-section explain-disagreements" id="explainDisagreementsSection" style="display:none;">
                    <h4>Model Disagreements (flagged conflicts)</h4>
                    <ul id="explainVisualDisagreements"></ul>
                </div>
                <div class="explain-section explain-recommendation">
                    <h4>Recommendation</h4>
                    <p id="explainRecommendation">--</p>
                </div>
                <div class="explain-section">
                    <h4>Classifier Agreement</h4>
                    <p id="explainAgreement">--</p>
                </div>
                <div class="explain-section explain-llm-passes">
                    <h4>LLM Pass Status</h4>
                    <div id="explainLlmPasses" class="llm-passes-grid"></div>
                </div>
                <div class="explain-section explain-quality" id="explainQualitySection" style="display:none;">
                    <h4>Quality Warnings</h4>
                    <ul id="explainQualityWarnings"></ul>
                </div>
                <div class="explain-section explain-disclaimer">
                    <h4>Disclaimer</h4>
                    <p id="explainDisclaimer">Not a medical diagnosis. Research / educational only.</p>
                </div>
                <details class="explain-raw">
                    <summary>Polished summary (verified prose, may equal Impression if LLM polish rejected)</summary>
                    <p id="explainSummary"></p>
                </details>
                <details class="explain-raw">
                    <summary>Raw deterministic features (JSON)</summary>
                    <pre id="explainRaw"></pre>
                </details>
            </div>
        `;
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
            // Persist for the Export Report download.
            this.currentExplanation = payload.explanation || null;
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

        // --- Classifier-negative explanation (shown only when verdict=no_tumor) ---
        const negativeSection = document.getElementById('explainNegativeSection');
        const negativeText = document.getElementById('explainNegativeExplanation');
        const negExp = exp.classifier_negative_explanation;
        if (negativeSection && negativeText) {
            if (negExp) {
                negativeSection.style.display = 'block';
                negativeText.textContent = negExp;
            } else {
                negativeSection.style.display = 'none';
                negativeText.textContent = '';
            }
        }

        // --- Pattern D: vision LLM reasoning for negative cases -----------
        const visionNegSection = document.getElementById('explainVisionNegativeSection');
        const visionNegText = document.getElementById('explainVisionNegativeText');
        const visionNeg = exp.vision_negative_reasoning;
        if (visionNegSection && visionNegText) {
            if (visionNeg) {
                visionNegSection.style.display = 'block';
                visionNegText.textContent = visionNeg;
            } else {
                visionNegSection.style.display = 'none';
                visionNegText.textContent = '';
            }
        }

        const findingDomains = [
            ['geometry', 'Geometry'],
            ['localization', 'Localization'],
            ['intensity', 'Intensity'],
            ['texture', 'Texture'],
            ['multimodal', 'Multimodal'],
            ['morphology_margins', 'Morphology & Margins'],
            ['internal_architecture', 'Internal Architecture'],
            ['mass_effect', 'Mass Effect'],
        ];
        const renderFindingsInto = (el, fmap) => {
            if (!el || !fmap) return false;
            const html = findingDomains
                .filter(([k]) => fmap[k])
                .map(([k, label]) => `<dt>${label}</dt><dd>${this.escapeHtml(fmap[k])}</dd>`)
                .join('');
            el.innerHTML = html;
            return Boolean(html);
        };

        // --- Structured findings (8 domains, primary report) --------------
        const findingsEl = document.getElementById('explainFindings');
        const findingsSection = document.getElementById('explainFindingsSection');
        const hasFindings = renderFindingsInto(findingsEl, exp.findings || {});
        // Hide the whole section when the deterministic narrative produced no
        // primary findings (i.e. verdict was no_tumor and the U-Net features
        // got moved into the FP-region debug bucket below).
        if (findingsSection) findingsSection.style.display = hasFindings ? '' : 'none';

        // --- Grade evidence narrative -------------------------------------
        const gradeEl = document.getElementById('explainGradeEvidence');
        const gradeSection = document.getElementById('explainGradeSection');
        const gradeText = exp.grade_evidence_narrative || '';
        if (gradeEl) gradeEl.textContent = gradeText || '--';
        if (gradeSection) gradeSection.style.display = gradeText ? '' : 'none';

        // --- False-positive region debug (collapsible) --------------------
        // When the classifier verdict was no_tumor, the U-Net feature
        // breakdown was preserved here under a clearly-labeled details block
        // so the raw data is accessible but not pretending to be a clinical
        // finding.
        const fpSection = document.getElementById('explainFpRegionSection');
        const fpFindings = document.getElementById('explainFpRegionFindings');
        const fpGrade = document.getElementById('explainFpRegionGrade');
        const fpAnalysis = exp.fp_region_analysis || null;
        const fpGradeText = exp.fp_grade_evidence || '';
        const hasFpAnalysis = renderFindingsInto(fpFindings, fpAnalysis || {});
        if (fpGrade) fpGrade.textContent = fpGradeText;
        if (fpSection) fpSection.style.display = (hasFpAnalysis || fpGradeText) ? '' : 'none';

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
            ['vizImage', 'heatmapImage', 'overlayImage', 'maskImage', 'segoverlayImage', 'vizPlaceholder'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });
            // Suppression banner is shown only on the U-Net Mask / U-Net
            // Overlay tabs when the classifier consensus is no-tumor.
            const banner = document.getElementById('maskSuppressBanner');
            if (banner) {
                const isSegTab = (tabType === 'mask' || tabType === 'segoverlay');
                if (this._maskSuppressed && isSegTab) {
                    banner.style.display = 'block';
                    banner.textContent = this._maskSuppressedReason || '';
                } else {
                    banner.style.display = 'none';
                }
            }
            const idMap = {
                original: 'vizImage',
                heatmap: 'heatmapImage',
                overlay: 'overlayImage',
                mask: 'maskImage',
                segoverlay: 'segoverlayImage',
            };
            const targetId = idMap[tabType] || `${tabType}Image`;
            const target = document.getElementById(targetId);
            // If the user picked a Grad-CAM tab and we have no real heatmap
            // (ONNX-only deploy, ViT hybrid, etc.), show the unavailable
            // placeholder instead of an empty image element.
            const needsAvailability = (tabType === 'heatmap' || tabType === 'overlay');
            if (needsAvailability && target && target.dataset.available === 'false') {
                const placeholder = document.getElementById('vizPlaceholder');
                if (placeholder) {
                    placeholder.style.display = 'flex';
                    placeholder.textContent = this._gradcamUnavailableReason
                        || 'Grad-CAM unavailable for the selected model.';
                }
                return;
            }
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
    
    // Inline toast for real informational events (e.g. "report exported").
    // Replaces the previous "Coming Soon" placeholder which advertised
    // unimplemented features.
    showToast(title, description, level = 'info') {
        const toast = document.createElement('div');
        toast.className = `nl-toast nl-toast-${level}`;
        toast.innerHTML = `
            <strong class="nl-toast-title">${this.escapeHtml(title)}</strong>
            <p class="nl-toast-desc">${this.escapeHtml(description || '')}</p>
        `;
        document.body.appendChild(toast);
        setTimeout(() => { toast.classList.add('nl-toast-exit'); }, 3500);
        setTimeout(() => { toast.remove(); }, 4000);
    }
    
    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }
    
    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
    
    /**
     * Export the analysis as JSON. Includes the classifier results, the cascade
     * segmentation decision, the full explanation payload (impression,
     * structured findings, grade evidence, differential with citations,
     * LLM-pass status), and the raw measured features. Sufficient to
     * reproduce the on-screen report from the file alone.
     */
    exportReport() {
        if (!this.currentResults) {
            this.showToast('No analysis to export', 'Run an analysis first.', 'error');
            return;
        }
        const report = {
            schema_version: '2.1',
            patient_id: this.currentResults.patientId,
            timestamp: this.currentResults.timestamp,
            diagnosis: this.currentResults.diagnosis,
            confidence: this.currentResults.confidence,
            best_model: this.currentResults.bestModel?.modelLabel,
            processing_time_seconds: this.currentResults.processingTime,
            model_results: this.currentResults.models,
            segmentation: this.currentSegmentation?.result || null,
            explanation: this.currentExplanation || null,
        };
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `neurolens_${this.currentResults.patientId}.json`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('Report exported', `${a.download} downloaded.`, 'success');
    }

    /**
     * Open the browser print dialog scoped to the result panel.
     * The print stylesheet hides the chrome (sidebar, top bar, controls,
     * raw-features blob) and prints just the radiology-style report. The
     * user picks "Save as PDF" in the print dialog for a portable file.
     */
    printReport() {
        if (!this.currentResults) {
            this.showToast('No analysis to print', 'Run an analysis first.', 'error');
            return;
        }
        window.print();
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

    /**
     * Live /status polling: server returns real ONNX session count, GPU
     * memory, LLM backend availability. Replaces the previous hard-coded
     * "3/3 models, 4.2/8 GB, 2 pending" mock that was misleading.
     */
    async loadStatus() {
        const list = document.getElementById('systemStatusList');
        const lastUpdated = document.getElementById('statusLastUpdated');
        try {
            const r = await fetch('/status', { headers: { 'Accept': 'application/json' } });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const s = await r.json();
            const rows = [];

            // Inference runtime row.
            const ort = s.onnx_runtime || {};
            const ortOk = !!ort.available;
            const provider = (ort.providers || []).find(p => p.includes('CUDA')) ? 'CUDA' :
                             (ort.providers || []).find(p => p.includes('CPU')) ? 'CPU' : '-';
            rows.push(`
                <div class="status-item">
                    <span class="status-dot ${ortOk ? 'online' : 'offline'}"></span>
                    <span>Inference Runtime</span>
                    <span class="status-value">${ortOk ? `ONNX ${provider}` : 'PyTorch'}</span>
                </div>`);

            // Loaded sessions
            rows.push(`
                <div class="status-item">
                    <span class="status-dot online"></span>
                    <span>Loaded Sessions</span>
                    <span class="status-value">${ort.sessions_loaded ?? 0}</span>
                </div>`);

            // GPU memory (only when actually available)
            const gpu = s.gpu || {};
            if (gpu.available) {
                const usedGb = ((gpu.memory_used_mb || 0) / 1024).toFixed(1);
                const totalGb = ((gpu.memory_total_mb || 0) / 1024).toFixed(1);
                const pct = gpu.memory_total_mb ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0;
                rows.push(`
                    <div class="status-item">
                        <span class="status-dot ${pct < 80 ? 'online' : 'warning'}"></span>
                        <span title="${this.escapeHtml(gpu.name || 'GPU')}">GPU Memory</span>
                        <span class="status-value">${usedGb} / ${totalGb} GB</span>
                    </div>`);
            } else {
                rows.push(`
                    <div class="status-item">
                        <span class="status-dot warning"></span>
                        <span>GPU</span>
                        <span class="status-value">CPU mode</span>
                    </div>`);
            }

            // Classifier weight readiness (count of present .onnx / .pt)
            const cls = s.classifiers || {};
            const clsCount = Object.values(cls).filter(c => c && (c.onnx || c.pt)).length;
            rows.push(`
                <div class="status-item">
                    <span class="status-dot ${clsCount >= 3 ? 'online' : 'warning'}"></span>
                    <span>Classifiers Ready</span>
                    <span class="status-value">${clsCount} / 3</span>
                </div>`);

            // Segmentation
            const segs = s.segmentation_models || [];
            const segCount = segs.filter(m => m.onnx || m.pt_size_mb).length;
            rows.push(`
                <div class="status-item">
                    <span class="status-dot ${segCount > 0 ? 'online' : 'offline'}"></span>
                    <span>Segmentation</span>
                    <span class="status-value">${segCount} model${segCount === 1 ? '' : 's'}</span>
                </div>`);

            // LLM backend availability
            const llm = s.llm || {};
            let llmStatus = 'deterministic only';
            let llmDot = 'warning';
            if (llm.hf_inference_token_present) { llmStatus = 'HF Inference'; llmDot = 'online'; }
            else if (llm.anthropic_token_present) { llmStatus = 'Anthropic'; llmDot = 'online'; }
            rows.push(`
                <div class="status-item">
                    <span class="status-dot ${llmDot}"></span>
                    <span>LLM Explanation</span>
                    <span class="status-value">${llmStatus}</span>
                </div>`);

            if (list) list.innerHTML = rows.join('');
            if (lastUpdated) {
                const t = new Date();
                lastUpdated.textContent = `updated ${t.getHours().toString().padStart(2,'0')}:${t.getMinutes().toString().padStart(2,'0')}`;
            }
        } catch (err) {
            if (list) {
                list.innerHTML = `
                    <div class="status-item">
                        <span class="status-dot offline"></span>
                        <span>Server Unreachable</span>
                        <span class="status-value">--</span>
                    </div>`;
            }
        }
    }

    /**
     * Session-scoped Recent Scans: pushes each finished analysis into the
     * sidebar list. Survives only as long as the tab is open (no persistence)
     * to keep the demo simple and avoid the misleading mock that was here.
     */
    addRecentScan(entry) {
        if (!this._recentScans) this._recentScans = [];
        // Attach a snapshot of the current view state so clicking the item
        // later can restore the full Results page (currentResults +
        // currentSegmentation + currentExplanation + imageDataUrl). For
        // batch entries the scan id starts with 'BATCH-' and is restored
        // via viewBatchEntry() which already does the lookup.
        if (!entry.snapshot && !String(entry.id || '').startsWith('BATCH-')) {
            entry.snapshot = {
                currentResults: this.currentResults,
                currentSegmentation: this.currentSegmentation,
                currentExplanation: this.currentExplanation || null,
                imageDataUrl: this.imageDataUrl,
                currentFile: this.currentFile,
            };
        }
        this._recentScans.unshift(entry);
        if (this._recentScans.length > 8) this._recentScans.length = 8;
        this.renderRecentScans();
    }

    renderRecentScans() {
        const el = document.getElementById('recentScansList');
        if (!el) return;
        const items = this._recentScans || [];
        if (!items.length) {
            el.innerHTML = '<div class="recent-empty">No scans yet. Upload an MRI to begin.</div>';
            return;
        }
        el.innerHTML = items.map(s => {
            const tumor = s.isPositive;
            const ago = this.formatRelativeTime(s.timestamp);
            return `
                <div class="recent-item" data-scan-id="${this.escapeHtml(s.id)}" role="button" tabindex="0" title="Click to reopen this analysis">
                    <div class="recent-icon ${tumor ? 'tumor' : 'normal'}">${tumor ? 'T' : 'N'}</div>
                    <div class="recent-info">
                        <span class="recent-id">${this.escapeHtml(s.id)}</span>
                        <span class="recent-time">${ago}</span>
                    </div>
                    <span class="recent-status ${tumor ? 'positive' : 'negative'}">${tumor ? 'Tumor' : 'Normal'}</span>
                </div>`;
        }).join('');

        // Bind clicks. We rebuild on every render so old handlers are
        // dropped along with the old DOM.
        el.querySelectorAll('.recent-item').forEach(node => {
            const scanId = node.dataset.scanId;
            const handler = () => this.viewRecentScan(scanId);
            node.addEventListener('click', handler);
            node.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handler();
                }
            });
        });
    }

    /** Reopen a previously-analysed scan from the Recent Scans sidebar. */
    viewRecentScan(scanId) {
        if (!scanId) return;
        // Batch entries restore through the existing viewBatchEntry path so
        // we don't duplicate the snapshot-restore logic.
        if (String(scanId).startsWith('BATCH-')) {
            return this.viewBatchEntry(scanId);
        }
        const entry = (this._recentScans || []).find(s => s.id === scanId);
        if (!entry || !entry.snapshot) {
            this.showToast('Snapshot unavailable',
                'This scan was added before the snapshot feature was wired up. Re-run the analysis to reopen it.', 'error');
            return;
        }
        const snap = entry.snapshot;
        if (snap.currentResults) this.currentResults = snap.currentResults;
        if (snap.currentSegmentation) this.currentSegmentation = snap.currentSegmentation;
        if (snap.currentExplanation !== undefined) this.currentExplanation = snap.currentExplanation;
        if (snap.imageDataUrl) this.imageDataUrl = snap.imageDataUrl;
        if (snap.currentFile) this.currentFile = snap.currentFile;
        if (this.currentResults) {
            this.displayResults();
        }
    }

    formatRelativeTime(ms) {
        const diff = Date.now() - ms;
        if (diff < 60_000) return 'just now';
        if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} min ago`;
        return `${Math.floor(diff / 3_600_000)} h ago`;
    }

    setupNavigation() {
        // Session-tracked Recent Scans bind themselves in addRecentScan().
        // No mock click handlers needed; the items appear only after real runs.
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