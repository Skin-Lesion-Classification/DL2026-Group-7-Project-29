# Skin-Lesion Classification under Severe Class Imbalance (HAM10000)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Deep Learning system for skin lesion classification across 7 dermatological disease categories (using the HAM10000 dataset), addressing severe class imbalance (where the majority class `nv` accounts for 66.95% and the minority class `df` only 1.15%).

---

## 📌 Project Overview
- **Objective**: Build a ResNet-50 model for dermatoscopic image classification, systematically experimenting with and comparing techniques to resolve class imbalance, focusing on two key metrics: **Macro F1** and **Balanced Accuracy**.
- **Zero Data Leakage**: Split Train / Val / Test sets (70% - 15% - 15%) based on `lesion_id` using `StratifiedGroupKFold`. This ensures images of the same lesion from the same patient never overlap across splits.
- **5 Ablation Configurations**:
  1. **Baseline**: Cross-Entropy Loss + Random Sampling + Basic Augmentation.
  2. **Sampling Ablation**: Weighted Random Sampler (inverse class frequency).
  3. **Loss Function Ablation**: Focal Loss ($\gamma = 2.0$).
  4. **Data Augmentation Ablation**: Color Jitter + Random Affine + Rotation.
  5. **Combined Strategy**: Combination of Weighted Sampling + Focal Loss + Data Augmentation.

---

## 📂 Project Structure
```text
skin-lesion-classification/
├── data/                       # HAM10000 image data and metadata files
│   ├── HAM10000_metadata.csv   # Clinical metadata (lesion_id, dx, age, sex, etc.)
│   ├── splits.csv              # Standard split file (Zero-Leakage Grouped Split)
│   └── images/                 # 10,015 dermatoscopic JPG images
├── models/
│   ├── __init__.py
│   └── resnet.py               # ResNet-50 network architecture for 7-class classification
├── results/                    # Model weight checkpoints and reports
│   ├── baseline/best_model.pth
│   ├── weighted_sampling/best_model.pth (Best Model Checkpoint)
│   ├── focal_loss/best_model.pth
│   ├── augmentation/best_model.pth
│   ├── combined/best_model.pth
│   ├── comparison.csv          # Summary table comparing all 5 configurations
│   ├── comparison_chart.png    # Visual bar chart comparing metrics
│   ├── per_class_recall_comparison.png # Recall comparison on rare classes
│   ├── confusion_matrix_test.png       # Confusion matrix on the Test set
│   ├── report.pdf              # Detailed 11-page academic report
│   └── ppt.pptx                # 4-slide structured presentation deck
├── data.py                     # Dataset management, transforms, Grouped K-Fold split
├── losses.py                   # Implementations of Cross-Entropy, Focal Loss, Class-Balanced Loss
├── train.py                    # Training pipeline, scheduler, checkpointing
├── test.py                     # Official testing script on the 1,431 test images
├── predict.py                  # Direct single-image inference script (no web server needed)
├── main.py                     # Main execution entrypoint (supports direct prediction & testing)
├── experiments.py              # Automated runner for benchmark evaluation
├── DATA.md                     # Detailed data specification and preparation pipeline docs
├── requirements.txt            # Python library dependencies
└── README.md                   # User guide and overview report
```

---

## ⚙️ Environment Setup

```bash
# 1. Activate Python environment (Python 3.10+)
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Test Model on Full Test Set (Standard Benchmark)
Run evaluation of the best model (`Weighted Sampling`) across all 1,431 independent test images (zero data leakage). This automatically prints the summary metrics table, per-class details, and saves the confusion matrix chart:
```bash
python test.py
```
*(Optional: specify another checkpoint to compare configurations)*:
```bash
python test.py --checkpoint results/baseline/best_model.pth
```

### 2. Direct Skin Lesion Prediction (No Web Server Required)
Input the path to any JPG image file, and the script will immediately preprocess it, calculate probability distributions across all 7 disease categories, and output the top prediction with its clinical risk level:
```bash
# Predict a specific image:
python predict.py --image data/images/ISIC_0024306.jpg

# Or quickly test a random image from the dataset:
python predict.py
```

### 3. Quick Execution via `main.py`
```bash
# Predict an image:
python main.py --image data/images/ISIC_0026273.jpg

# Or run test set evaluation:
python main.py --test
```

---

## 📈 Empirical Benchmark Results

Extracted directly from actual trained checkpoint outputs (`results/comparison.csv`):

| Experimental Configuration | Loss Function | Sampling Strategy | Augmentation Technique | Accuracy | Balanced Acc | Macro F1 (Key Focus) | Weighted F1 | Macro Recall |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Cross-Entropy | Random | Basic Flip | 83.23% | 58.29% | 0.6197 | 0.8221 | 58.29% |
| **Weighted Sampling** | **Cross-Entropy** | **Weighted Random** | **Basic Flip** | **78.48%** | **68.86%** | **0.6905** | **0.7996** | **68.86%** |
| **Focal Loss** | Focal ($\gamma=2$) | Random | Basic Flip | 79.59% | 65.64% | 0.6550 | 0.8028 | 65.64% |
| **Data Augmentation** | Cross-Entropy | Random | Color + Affine | 81.48% | 57.59% | 0.5922 | 0.8033 | 57.59% |
| **Combined Strategy** | Focal ($\gamma=2$) | Weighted Random | Color + Affine | 68.48% | 67.61% | 0.5845 | 0.7157 | 67.61% |

### Key Experimental Insights:
1. **The Traditional Accuracy Trap**: The Baseline configuration achieved high Accuracy (83.23%) mostly by overfitting to the majority class (`nv` representing 67% of data). Its Balanced Accuracy reached only 58.29%, missing many malignant cancer cases (`mel`, `bcc`).
2. **Superior Performance of Weighted Sampling**: Achieved the **highest Macro F1 (0.6905)** and **highest Balanced Accuracy (68.86%)**, improving sensitivity for detecting basal cell carcinoma (`bcc`) to **79.73%** and melanoma (`mel`) to **66.67%**.

---

## 📄 Reports & Presentation Slides
- **Monograph Report (PDF)**: `results/report.pdf` (11 pages, complete with mathematical formulas, comparison tables, and clinical analysis).
- **Presentation Slides (PPTX)**: `results/ppt.pptx` (4 slides structured for project defense).