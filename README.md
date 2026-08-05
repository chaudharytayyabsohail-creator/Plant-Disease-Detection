# LeafGuard AI - Plant Disease Detection System

Welcome to the LeafGuard AI project repository. This system is a deep-learning-based plant disease detection platform designed to assist farmers, growers, and agronomists. It features image classification, Grad-CAM model explainability overlays, and instant treatment recommendations.

This repository hosts implementation milestones for the project.

---

## Directory Structure

```text
leafguard_ai/
│
├── data/
│   └── plantvillage_metadata.csv   # Structured metadata of PlantVillage images with quality metrics
│
├── class_distribution.png          # Visual plot of crop and disease categories distribution
├── image_augmentation.png          # Visual demonstration of raw image rotation & flipping
├── sample_leaf.jpg                 # Simulated leaf image used for testing augmentations
├── confusion_matrices.png          # Confusion Matrix comparisons for ML classifiers
├── model_comparison.png            # Bar chart comparing Accuracy, Precision, Recall, F1
│
├── preprocessing.ipynb             # Phase 1: Tabular and CV image preprocessing pipeline
├── eda_feature_engineering.ipynb   # Phase 2: Feature engineering and visual EDA profiling
├── model_training_evaluation.ipynb # Phase 3: Model training, evaluation, and cross-validation
├── app.py                          # Streamlit application UI & frontend dashboard (FR-09 reports)
├── report.pdf                      # Phase 1 PDF Report (Milestone 1)
├── requirements.txt                # Python libraries locked dependencies
│
├── src/                            # Modular Python backend scripts
│   ├── __init__.py
│   ├── preprocessing.py            # Preprocessing cleaning & real-time image quality checks
│   ├── features.py                 # Feature engineering attribute equations
│   └── predict.py                  # Predictor class running model inference
│
└── README.md                       # Documentation and project walkthrough
```

---

## Milestone 1: Dataset Selection & Preprocessing
*   **Preprocessing Pipeline:** Drop duplicates, impute missing values using median/mode, cap outliers using Interquartile Range (IQR) on exposure/contrast metrics, apply `StandardScaler` scaling, encode features (Label Encoding and One-Hot Encoding), and handle tabular class imbalance using `SMOTE-Tomek`.
*   **Jupyter Notebook:** [preprocessing.ipynb](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/preprocessing.ipynb)
*   **Report Summary:** [report.pdf](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/report.pdf)

---

## Milestone 2: Feature Engineering & Exploratory Data Visualization
Focuses on creating rich attributes and visually profiling relationships within our leaf metadata prior to model development.
*   **Feature Engineering:** Created `resolution` (width * height), `aspect_ratio`, `density_score` (file size / resolution), and `brightness_blur_interaction`.
*   **Feature Selection:** Dropped `height` due to perfect multicollinearity (1.00 correlation with `width`). Calculated Mutual Information (MI) scores to rank feature importance.
*   **Jupyter Notebook:** [eda_feature_engineering.ipynb](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/eda_feature_engineering.ipynb)

---

## Milestone 3: Model Training & Evaluation
Focuses on building classification models, validating their performance, and choosing the optimal classifier.
*   **Algorithms Evaluated:** Trained and compared **Logistic Regression**, **Random Forest Classifier**, and **Support Vector Machine (SVM)**.
*   **Robustness Checking:** Applied **5-Fold Cross-Validation** on the training set to prevent overfitting.
*   **Jupyter Notebook:** [model_training_evaluation.ipynb](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/model_training_evaluation.ipynb)

---

## Milestone 4 & 5: ML Project Deployment (Phase 1 & 2)
Focuses on packaging the model into a modular, production-ready web application.

### 1. Serialized Binaries (`models/`)
*   `leafguard_model.joblib`: Serialized Random Forest Classifier weights.
*   `scaler.joblib`: Serialized StandardScaler.
*   `feature_names.joblib`: Column template.

### 2. Modular Backend Scripts (`src/`)
*   `src/preprocessing.py`: Real-time image quality inspection checks (exposure range, Laplacian variance blur threshold, size limits).
*   `src/features.py`: Computes engineered dimensions dynamically.
*   `src/predict.py`: Reindexes one-hot encoded crop features, scales variables, executes model decision trees, and returns status outputs with treatment advisories.

### 3. Streamlit Dashboard App (`app.py`)
*   **Quality Metrics Indicator**: Computes file size, exposure, and sharpness dynamically upon file upload.
*   **Model Prediction**: Outputs Healthy vs. Diseased leaf status with prediction confidence.
*   **XAI Heatmap**: Toggles mock Grad-CAM overlay to highlight infection spots.
*   **Treatment Cards**: Renders organic and chemical advisory guides.
*   **Diagnostic Report Export (FR-09)**: Interactive sidebar history list that allows users to export and download their diagnostic logs as a CSV report sheet.

---

## How to Setup and Run

### 1. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Start the Streamlit server locally:
```bash
streamlit run app.py
```
This will open the dashboard in your default browser at `http://localhost:8501`.
