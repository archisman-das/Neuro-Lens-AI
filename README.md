# Explainable Vision Transformer-Based Brain Tumor Detection

This project implements the roadmap from the attached Word documents:
- Baseline CNN model
- Transfer learning with ResNet50
- Vision Transformer model
- Explainability using Grad-CAM and ViT attention visualization
- Comparative evaluation across accuracy, precision, recall, F1-score, and ROC-AUC

## Structure
- `src/data.py`: Dataset loading, preprocessing, augmentation
- `src/models.py`: Baseline CNN, transfer learning, and ViT model builders
- `src/train.py`: Training workflow with configurable model selection
- `src/evaluate.py`: Evaluation and metrics reporting
- `src/explain.py`: Grad-CAM and attention explainability outputs
- `requirements.txt`: Python dependencies

## Dataset
Place the Kaggle Brain MRI dataset in `dataset/` with subfolders:
- `dataset/train/tumor`
- `dataset/train/no_tumor`
- `dataset/val/tumor`
- `dataset/val/no_tumor`
- `dataset/test/tumor`
- `dataset/test/no_tumor`

If you only have a single split, use `src/data.py` to generate train/val/test splits.

## Quick start
1. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Prepare the dataset
   - Place the raw Kaggle Brain MRI images in `dataset/raw/<class-name>`.
   - Use the split helper if your dataset is not already separated:
     ```powershell
     python src\split_dataset.py --source dataset\raw --output dataset
     ```
4. Train a baseline CNN:
   ```powershell
   python src\train.py --model cnn
   ```
5. Evaluate a trained model:
   ```powershell
   python src\evaluate.py --model cnn --weights artifacts\cnn\best_weights.weights.h5
   ```
6. Generate explainability outputs:
   ```powershell
   python src\explain.py --model cnn --weights artifacts\cnn\best_weights.weights.h5
   ```
7. Run the dashboard:
   ```powershell
   streamlit run app.py
   ```

## Documentation
A full project overview, architecture, and result summary are available in `PROJECT_DOCUMENTATION.md`.

## License
This project is released under the MIT License. See `LICENSE` for details.
