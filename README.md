# LeafGuard AI - Plant Disease Detection System

## Phase 1: Dataset Selection & Data Preprocessing

LeafGuard AI is a computer vision platform designed to assist farmers, growers, and agronomists by providing real-time, image-based plant disease classification alongside Grad-CAM explanations and preventive treatment recommendations.

This repository covers **Phase 1: Dataset Selection & Data Preprocessing**, focusing on preparing a clean, machine learning-ready pipeline using the PlantVillage and PlantDoc datasets.

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
├── preprocessing.ipynb             # Fully executed Jupyter Notebook with tabular and CV preprocessing
├── report.pdf                      # One-page PDF Project Report documenting Phase 1 outcomes
└── README.md                       # Repository instructions and overview
```

---

## Dataset Selection & Understanding

*   **Primary Dataset (PlantVillage):** Consists of 54,305 high-quality leaf images across 14 crop species and 38 classes under lab-controlled conditions.
*   **Secondary Dataset (PlantDoc):** Consists of 2,598 field-captured leaf images across 13 species and 17 disease classes, used to validate real-world generalization and prevent background/lighting bias.
*   **Metadata Representation (`data/plantvillage_metadata.csv`):** Contains tabular features corresponding to image file size, height, width, brightness, and Laplacian blurriness scores. Includes injected missing values and duplicates to demonstrate standard tabular data cleaning.

---

## Preprocessing Pipeline

1.  **Duplicate Removal:** Dropped 50 duplicate records from the dataset metadata.
2.  **Missing Value Imputation:** Numerical features (`file_size_kb`, `brightness`) are imputed using their medians, and categorical features (`crop_type`) are imputed using their mode.
3.  **Outlier Capping (IQR):** Image properties are clipped within `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]` bounds to mitigate extremely dark, overexposed, or blurry images from skewing the distributions.
4.  **Categorical Encoding:** Target label `status` is converted to binary using Label Encoding (1=Healthy, 0=Diseased). Feature `crop_type` is encoded using One-Hot Encoding.
5.  **Feature Scaling:** Standardized continuous features using `StandardScaler` to bring them to a mean of 0 and standard deviation of 1.
6.  **Class Balancing (SMOTE-Tomek):** Imbalance is analyzed (898 Healthy vs 1102 Diseased). SMOTE-Tomek is applied to balance the tabular metadata representation. (An explanation is provided on why SMOTE is not used for raw pixels, which instead use data augmentation).
7.  **Computer Vision Pipeline:** Demonstration of image loading, resizing to $256 \times 256$, pixel normalization ($1/255$), and data augmentation (rotations, flips).

---

## Technical Stack & Libraries

*   **Language:** Python 3.12
*   **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (`SMOTE-Tomek`), Matplotlib, Seaborn, ReportLab (for PDF generation), nbformat.
*   **Tools:** Jupyter Notebook, VS Code, Git.

---

## How to Run the Preprocessing Notebook

### 1. Prerequisites
Ensure Python 3.12+ is installed on your system.

### 2. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn notebook
```

### 3. Open and Run Notebook
Launch Jupyter Notebook in the project directory:
```bash
jupyter notebook
```
Double-click on `preprocessing.ipynb` and click **Run All** cells.
