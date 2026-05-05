# Cluster Analysis and Predictive Models for Biomarker Identification

**Bachelor's Thesis — University of Milan, AnacletoLAB**  
Computational Biology and Bioinformatics Lab

---

## Overview

This project applies machine learning to behavioral data from murine models to support biomarker identification in a biomedical research context. The pipeline covers feature selection via correlation analysis, multi-model comparison across sex-stratified groups, and interpretability analysis using SHAP values.

The dataset includes 15 behavioral features measured in male and female mice under stress conditions. After feature selection, 11 features were retained for males and 10 for females.

---

## Pipeline

### 1. `corr-analisi-prova.py` — Feature Selection

- Computes a Pearson correlation matrix across all behavioral features
- Removes features with |r| > 0.8 to reduce collinearity
- **Males:** removed `tOP`, `t%OP` → 11 features retained
- **Females:** removed `tOP`, `% OP`, `t%OP` → 10 features retained
- Outputs selected feature CSVs for downstream analysis

### 2. `multimodello.py` — Multi-Model Comparison

Trains and evaluates four supervised learning algorithms on sex-stratified groups (males, females) using cross-validation:

- Random Forest (RF)
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- XGBoost

### 3. `final-shap.py` — SHAP Interpretability & Permutation Importance

- Computes SHAP values (global and per-sample) for RF and LR
- Generates beeswarm, bar, and waterfall SHAP plots
- Compares SHAP importance vs. permutation importance (F1, Precision, Recall, AUPRC)
- Produces forest plots and cross-method comparison charts

---

## Results

### Males

| Model | Accuracy | F1 | Recall | Precision | ROC AUC | AUPRC |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.875 | **0.909** | **0.938** | 0.882 | 0.883 | 0.940 |
| Logistic Regression | 0.875 | 0.903 | 0.875 | **0.933** | **0.953** | **0.979** |
| SVM | 0.792 | 0.839 | 0.813 | 0.867 | 0.359 | 0.593 |
| XGBoost | 0.792 | 0.849 | 0.875 | 0.824 | 0.762 | 0.834 |

**Best models on males:** RF and LR tied on accuracy (0.875), RF slightly higher F1, LR higher AUC and AUPRC.

### Females

| Model | Accuracy | F1 | Recall | Precision | ROC AUC | AUPRC |
|---|---|---|---|---|---|---|
| Random Forest | 0.708 | 0.811 | **0.938** | 0.714 | 0.625 | 0.761 |
| Logistic Regression | 0.583 | 0.615 | 0.500 | **0.800** | 0.664 | **0.812** |
| SVM | 0.500 | 0.571 | 0.500 | 0.667 | 0.422 | 0.643 |
| **XGBoost** | **0.750** | **0.833** | **0.938** | 0.750 | 0.695 | 0.788 |

**Best model on females:** XGBoost (highest accuracy and F1). RF competitive on recall. Overall lower performance vs. males, consistent with higher behavioral variability in the female group.

---

## Key Findings — Feature Importance

### SHAP Analysis (Logistic Regression)

**Males** — top features by mean |SHAP|:
- `tCENT` (~5.0), `social interaction time` (~3.0), `tCL` (~1.2), `% OP` (~1.0)

**Females** — top features by mean |SHAP|:
- `social interaction time` (~5.3), `locomotor activity` (~3.4), `% OP` (~2.9), `tCENT` (~2.0), `body weight` (~2.0)

> **Notable sex difference:** `tCENT` dominates in males; `social interaction time` dominates in females. This suggests distinct behavioral biomarker profiles between sexes.

### Permutation Importance (F1-weighted average across groups)

`social interaction time` and `tCENT` are the two most impactful features overall for LR. RF shows much lower and more distributed importance scores, consistent with its ensemble nature.

---

## Visualizations

### SHAP — Females, Logistic Regression

**Global bar (mean |SHAP|)**

![SHAP bar females LR](esecuzione%201%20final/femmine_Logistic_Regression_shap_bar.png)

**Beeswarm (feature value vs. impact)**

![SHAP beeswarm females LR](esecuzione%201%20final/femmine_Logistic_Regression_shap_beeswarm.png)

**Waterfall — sample 0**

![SHAP waterfall females LR](esecuzione%201%20final/femmine_Logistic_Regression_shap_waterfall_sample0.png)

---

### SHAP — Males, Logistic Regression

**Global bar (mean |SHAP|)**

![SHAP bar males LR](esecuzione%201%20final/maschi_Logistic_Regression_shap_bar.png)

**Beeswarm**

![SHAP beeswarm males LR](esecuzione%201%20final/maschi_Logistic_Regression_shap_beeswarm.png)

**Waterfall — sample 0**

![SHAP waterfall males LR](esecuzione%201%20final/maschi_Logistic_Regression_shap_waterfall_sample0.png)

---

### Permutation Importance — RF vs LR

**Overall average — F1**

![Final average F1](esecuzione%201%20final/final_average_importance_f1_RF_vs_LR.png)

**Overall average — Precision**

![Final average Precision](esecuzione%201%20final/final_average_importance_precision_RF_vs_LR.png)

**Overall average — Recall**

![Final average Recall](esecuzione%201%20final/final_average_importance_recall_RF_vs_LR.png)

---

### Forest Plot — SHAP vs Permutation (Females, LR, F1)

![Forest plot females LR](esecuzione%201%20final/femmine_Logistic_Regression_forest_SHAP_vs_PERM_f1.png)

---

## Repository Structure

```
├── corr-analisi-prova.py       # Feature selection via correlation analysis
├── multimodello.py             # Multi-model training and evaluation
├── final-shap.py               # SHAP analysis and permutation importance
├── CMS-databehaviour_*.csv     # Raw behavioral dataset (males + females)
├── selected_features_*.csv     # Post-selection feature sets
├── *_shap_*.png                # SHAP plots (bar, beeswarm, waterfall)
├── *_importance_*.png          # Permutation importance plots
└── README.md
```

---

## Requirements

```
python >= 3.8
scikit-learn
xgboost
shap
pandas
numpy
matplotlib
seaborn
```

Install with:
```bash
pip install scikit-learn xgboost shap pandas numpy matplotlib seaborn
```

---

## Context

This project was developed as part of a Bachelor's thesis at the **AnacletoLAB — Computational Biology and Bioinformatics Lab**, University of Milan. The behavioral data comes from a murine model of chronic mild stress (CMS), a widely used paradigm for studying depression-related phenotypes. The goal is to identify behavioral biomarkers that can reliably distinguish stressed from control animals, with a sex-stratified analysis to account for known differences in stress response between males and females.ehavioral biomarkers that can reliably distinguish stressed from control animals, with a sex-stratified analysis to account for known differences in stress response between males and females.
