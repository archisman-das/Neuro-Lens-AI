# NeuroLens AI — React Component Hierarchy & Sample Props

Top-level App
- TopNav
  - props: { onSearch(patientId), onToggleTheme(), notifications: [], user }
- Sidebar
  - props: { menuItems: [] , collapsed: false }
- MainLayout
  - HeroSection (Upload)
    - props: { onUpload(file), supportedFormats: ['nii','dcm','png','jpg'] }
  - PredictionCardsRow
    - PredictionCard (Tumor Status)
      - props: { title: 'Tumor Status', value: 'Tumor', color: '--danger', icon }
    - PredictionCard (Tumor Type)
      - props: { title: 'Tumor Type', value: 'Glioma', topK: [{label,prob}] }
    - PredictionCard (Confidence)
      - props: { title: 'Confidence', value: 0.92, showLiveBar: true }
    - PredictionCard (Processing Time)
      - props: { title: 'Processing Time', valueMs: 1342, jobId }
  - VisualizationPanel
    - MRIViewer
      - props: { slices: [], currentIndex, onChangeSlice, zoom, onZoom }
    - OverlayControls
      - props: { showGradCAM, showViTAttention, opacity }
  - ModelAnalytics
    - MetricsCharts (Accuracy/Loss, ROC, ConfusionMatrix)
      - props: { metricsPayload, modelName }
  - PatientInfoPanel
    - props: { patient: {id,name,age,notes}, lastScan }
  - ActivityTimeline
    - props: { events: [{type,desc,timestamp,jobId}] }
  - Footer
    - props: { modelVersion, framework }

Notes
- Keep components small and prop-driven for easy testing and storybook usage.
- PredictionCard is a reusable component with `variant` prop to switch presentation.
