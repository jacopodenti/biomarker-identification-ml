# Cluster Analysis and Predictive Models for Biomarker Identification

Bachelor's thesis project — University of Milan, AnacletoLAB  
Computational Biology and Bioinformatics Lab

## Overview

This project applies machine learning to behavioral data from murine models 
to support biomarker identification in a biomedical research context.
The pipeline covers feature selection, multi-model comparison and 
interpretability analysis using SHAP values.

## Pipeline

### 1. `corr-analisi-prova.py` — Feature Selection
- Computes correlation matrix
- Removes features with |r| > 0.8 to reduce collinearity
- Outputs a list of selected independent variables

### 2. `multimodello.py` — Model Comparison
Trains and evaluates four supervised learning algorithms:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost

Evaluation metrics: Accuracy, F1-score, AUPRC  
Results are visualized as barcharts across male, female and overall groups.

### 3. `final-shap.py` — Full Pipeline + Interpretability
- Random Forest and Logistic Regression with repeated cross-validation (11 iterations)
- Permutation Importance and SHAP values
- Outputs: forest plots (RF vs LR), normalized SHAP vs Permutation plots, 
  individual SHAP waterfall plots
- Flag `USE_TEST_FOR_PERM` toggles computation on train or test set

## Requirements

Python ≥ 3.9

pip install numpy pandas matplotlib seaborn scikit-learn shap xgboost openpyxl
