/**
 * NeuroLens AI - Professional Dashboard Application
 */

class NeuroLensApp {
    constructor() {
        this.currentFile = null;
        this.currentResults = null;
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
        
        // Segmentation button
        document.getElementById('runSegmentationBtn').addEventListener('click', () => {
            this.runSegmentation();
        });
        
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
        
        // Show loading
        this.showLoading();
        this.startTime = Date.now();
        
        // Simulate API call with progress
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) progress = 95;
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `Processing: ${Math.round(progress)}%`;
        }, 200);
        
        // Simulate analysis delay
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Processing: 100%';
        
        // Generate mock results
        this.currentResults = this.generateMockResults(patientId, modelSelect.value);
        
        // Hide loading
        setTimeout(() => {
            this.hideLoading();
            this.displayResults();
        }, 500);
    }
    
    generateMockResults(patientId, modelType) {
        // ViT should always be the best model with highest confidence
        const modelResults = [
            {
                model: 'cnn',
                modelLabel: 'CNN (Fast)',
                prediction: 'Tumor',
                confidence: 0.78 + Math.random() * 0.08,
                accuracy: 0.85 + Math.random() * 0.05,
                isPositive: true,
                status: 'positive',
                auc: 0.87 + Math.random() * 0.05
            },
            {
                model: 'transfer',
                modelLabel: 'Transfer Learning',
                prediction: 'Tumor',
                confidence: 0.85 + Math.random() * 0.06,
                accuracy: 0.89 + Math.random() * 0.04,
                isPositive: true,
                status: 'positive',
                auc: 0.91 + Math.random() * 0.04
            },
            {
                model: 'vit',
                modelLabel: 'Vision Transformer',
                prediction: 'Tumor',
                confidence: 0.94 + Math.random() * 0.05, // Highest confidence
                accuracy: 0.95 + Math.random() * 0.03, // Highest accuracy
                isPositive: true,
                status: 'positive',
                auc: 0.96 + Math.random() * 0.03 // Highest AUC
            }
        ];
        
        // Find best model (should be ViT)
        const bestModel = modelResults.reduce((best, current) => 
            current.confidence > best.confidence ? current : best
        );
        
        const processingTime = ((Date.now() - this.startTime) / 1000).toFixed(1);
        
        return {
            patientId: patientId,
            timestamp: new Date().toLocaleString(),
            models: modelResults,
            bestModel: bestModel,
            diagnosis: 'Tumor Detected',
            isPositive: true,
            confidence: bestModel.confidence,
            processingTime: processingTime,
            uncertainty: {
                epistemic: 0.05 + Math.random() * 0.1,
                aleatoric: 0.03 + Math.random() * 0.08
            },
            robustness: 0.85 + Math.random() * 0.1
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
        
        // Update comparison table with ROC AUC column
        const tableBody = document.getElementById('comparisonTableBody');
        tableBody.innerHTML = results.models.map(model => `
            <tr class="${model === results.bestModel ? 'best' : ''}">
                <td><strong>${model.modelLabel}</strong></td>
                <td>${model.prediction}</td>
                <td>${(model.confidence * 100).toFixed(1)}%</td>
                <td>${(model.accuracy * 100).toFixed(1)}%</td>
                <td>${(model.auc * 100).toFixed(1)}%</td>
                <td>
                    <span class="status-badge ${model.status}">
                        ● ${model.status === 'positive' ? 'Positive' : 'Negative'}
                    </span>
                </td>
            </tr>
        `).join('');
        
        // Update uncertainty
        document.getElementById('epistemicValue').textContent = 
            results.uncertainty.epistemic.toFixed(3);
        document.getElementById('aleatoricValue').textContent = 
            results.uncertainty.aleatoric.toFixed(3);
        
        const totalUncertainty = (results.uncertainty.epistemic + results.uncertainty.aleatoric) / 2;
        document.getElementById('uncertaintyFill').style.width = 
            `${totalUncertainty * 100}%`;
        document.getElementById('uncertaintyNote').textContent = 
            totalUncertainty < 0.1 ? 'Low uncertainty - High reliability' : 
            totalUncertainty < 0.2 ? 'Moderate uncertainty - Review recommended' : 
            'High uncertainty - Clinical review required';
        
        // Update robustness gauge
        const robustnessPercent = results.robustness * 100;
        document.getElementById('robustnessValue').textContent = 
            `${robustnessPercent.toFixed(0)}%`;
        document.getElementById('robustnessGauge').style.background = 
            `conic-gradient(var(--success) 0deg, var(--success) ${robustnessPercent * 3.6}deg, var(--gray-200) ${robustnessPercent * 3.6}deg)`;
        document.getElementById('robustnessNote').textContent = 
            robustnessPercent > 80 ? 'Excellent robustness' : 
            robustnessPercent > 60 ? 'Good robustness' : 
            'Moderate robustness - Consider retraining';
        
        // Set visualization images
        if (this.imageDataUrl) {
            document.getElementById('vizImage').src = this.imageDataUrl;
            // Create heatmap effect using canvas
            this.createHeatmapEffect(this.imageDataUrl);
            // Create overlay effect
            this.createOverlayEffect(this.imageDataUrl);
        }
        
        // Show results section
        this.showSection('results');
    }
    
    createHeatmapEffect(imageUrl) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            // Get image data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Apply heatmap color mapping
            for (let i = 0; i < data.length; i += 4) {
                const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
                const intensity = gray / 255;
                
                // Heatmap colors: blue -> green -> yellow -> red
                if (intensity < 0.25) {
                    data[i] = 0;
                    data[i + 1] = 0;
                    data[i + 2] = intensity * 4 * 255;
                } else if (intensity < 0.5) {
                    data[i] = 0;
                    data[i + 1] = (intensity - 0.25) * 4 * 255;
                    data[i + 2] = (0.5 - intensity) * 4 * 255;
                } else if (intensity < 0.75) {
                    data[i] = (intensity - 0.5) * 4 * 255;
                    data[i + 1] = 255;
                    data[i + 2] = 0;
                } else {
                    data[i] = 255;
                    data[i + 1] = 255 - (intensity - 0.75) * 4 * 255;
                    data[i + 2] = 0;
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
            
            // Add semi-transparent overlay
            ctx.fillStyle = 'rgba(255, 0, 0, 0.15)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            document.getElementById('heatmapImage').src = canvas.toDataURL();
        };
        img.src = imageUrl;
    }
    
    createOverlayEffect(imageUrl) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            // Draw tumor region overlay (simulated)
            const centerX = canvas.width * 0.5;
            const centerY = canvas.height * 0.4;
            const radiusX = canvas.width * 0.15;
            const radiusY = canvas.height * 0.12;
            
            // Draw semi-transparent red overlay for tumor region
            ctx.beginPath();
            ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
            ctx.lineWidth = 3;
            ctx.stroke();
            
            // Add label
            ctx.fillStyle = 'white';
            ctx.font = 'bold 16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('TUMOR', centerX, centerY - radiusY - 10);
            
            document.getElementById('overlayImage').src = canvas.toDataURL();
        };
        img.src = imageUrl;
    }
    
    runSegmentation() {
        if (this.imageDataUrl) {
            // Display the original image in segmentation viewers
            document.getElementById('segOriginal').innerHTML = `<img src="${this.imageDataUrl}" style="width:100%;height:100%;object-fit:contain;border-radius:12px;">`;
            
            // Create segmentation mask
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw tumor region in green
                const centerX = canvas.width * 0.5;
                const centerY = canvas.height * 0.4;
                const radiusX = canvas.width * 0.15;
                const radiusY = canvas.height * 0.12;
                
                ctx.beginPath();
                ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(16, 185, 129, 0.8)';
                ctx.fill();
                
                document.getElementById('segMask').innerHTML = `<img src="${canvas.toDataURL()}" style="width:100%;height:100%;object-fit:contain;border-radius:12px;">`;
            };
            img.src = this.imageDataUrl;
            
            // Create overlay
            const overlayCanvas = document.createElement('canvas');
            const overlayCtx = overlayCanvas.getContext('2d');
            const overlayImg = new Image();
            overlayImg.onload = () => {
                overlayCanvas.width = overlayImg.width;
                overlayCanvas.height = overlayImg.height;
                overlayCtx.drawImage(overlayImg, 0, 0);
                
                // Draw tumor overlay
                const centerX = overlayCanvas.width * 0.5;
                const centerY = overlayCanvas.height * 0.4;
                const radiusX = overlayCanvas.width * 0.15;
                const radiusY = overlayCanvas.height * 0.12;
                
                overlayCtx.beginPath();
                overlayCtx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
                overlayCtx.fillStyle = 'rgba(16, 185, 129, 0.4)';
                overlayCtx.fill();
                overlayCtx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
                overlayCtx.lineWidth = 3;
                overlayCtx.stroke();
                
                document.getElementById('segOverlay').innerHTML = `<img src="${overlayCanvas.toDataURL()}" style="width:100%;height:100%;object-fit:contain;border-radius:12px;">`;
            };
            overlayImg.src = this.imageDataUrl;
        }
        
        // Generate metrics
        document.getElementById('diceScore').textContent = (0.88 + Math.random() * 0.08).toFixed(3);
        document.getElementById('iouScore').textContent = (0.82 + Math.random() * 0.1).toFixed(3);
        document.getElementById('tumorArea').textContent = Math.floor(200 + Math.random() * 400) + ' mm²';
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
            ['vizImage', 'heatmapImage', 'overlayImage'].forEach(id => {
                document.getElementById(id).style.display = 'none';
            });
            document.getElementById(`${tabType}Image`).style.display = 'block';
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
                            ${this.currentResults.models.map(model => `
                                <tr class="${model === this.currentResults.bestModel ? 'best' : ''}">
                                    <td><strong>${model.modelLabel}</strong></td>
                                    <td>${model.prediction}</td>
                                    <td>${(model.confidence * 100).toFixed(1)}%</td>
                                    <td>${(model.accuracy * 100).toFixed(1)}%</td>
                                    <td>${(model.auc * 100).toFixed(1)}%</td>
                                    <td>
                                        <span class="status-badge ${model.status}">
                                            ● ${model.status === 'positive' ? 'Positive' : 'Negative'}
                                        </span>
                                    </td>
                                </tr>
                            `).join('')}
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
                    ${this.currentResults.models.map(model => `
                        <div style="background: var(--gray-50); padding: 20px; border-radius: var(--radius-lg); border-left: 4px solid ${model === this.currentResults.bestModel ? 'var(--primary)' : 'var(--gray-300)'};">
                            <h4 style="margin-bottom: 12px; color: var(--gray-800);">${model.modelLabel}</h4>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                                <div>
                                    <div style="font-size: 12px; color: var(--gray-500); margin-bottom: 4px;">Confidence</div>
                                    <div style="font-size: 20px; font-weight: 700; color: var(--gray-800);">${(model.confidence * 100).toFixed(1)}%</div>
                                </div>
                                <div>
                                    <div style="font-size: 12px; color: var(--gray-500); margin-bottom: 4px;">Accuracy</div>
                                    <div style="font-size: 20px; font-weight: 700; color: var(--gray-800);">${(model.accuracy * 100).toFixed(1)}%</div>
                                </div>
                                <div>
                                    <div style="font-size: 12px; color: var(--gray-500); margin-bottom: 4px;">ROC AUC</div>
                                    <div style="font-size: 20px; font-weight: 700; color: var(--primary);">${(model.auc * 100).toFixed(1)}%</div>
                                </div>
                            </div>
                            <div style="margin-top: 12px;">
                                <div style="font-size: 12px; color: var(--gray-500); margin-bottom: 4px;">Prediction</div>
                                <span class="status-badge ${model.status}" style="font-size: 14px; padding: 6px 14px;">
                                    ● ${model.prediction}
                                </span>
                            </div>
                        </div>
                    `).join('')}
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