# Credit Default Prediction — Machine Learning Case Study

## Overview

This repository presents a supervised machine learning case study focused on predicting borrower payment difficulties using the publicly available Home Credit dataset.

The objective is to build a structured and leakage-safe modeling pipeline to estimate default risk at the loan application stage. The project emphasizes methodological rigor, disciplined validation, and systematic comparison of linear and non-linear models under class imbalance.

This repository contains the modeling and evaluation components developed by me.

---

## Problem Statement

Given applicant-level structured features (demographics, employment, financial variables, and credit history), predict whether a loan will experience payment difficulties (`TARGET = 1`).

The goals of this project are to:

- Build a robust classification pipeline  
- Compare multiple model families  
- Evaluate performance under severe class imbalance  
- Analyze trade-offs between recall and precision  
- Select and justify a final model  

---

## Dataset

- Source: Home Credit Default Risk (public dataset)
- Target:
  - `1` — Payment difficulties (default proxy)
  - `0` — No payment difficulties

The dataset is imbalanced (default rate ≈ 15%), making accuracy an unsuitable metric for model comparison.

---

## Methodology

### 1. Leakage-Safe Preprocessing

All preprocessing steps are implemented inside `Pipeline` and `ColumnTransformer` objects to ensure that transformations are learned **only on training folds** during cross-validation.

Key components include:

- **Custom imputation for `OWN_CAR_AGE`**
  - Non-owners → age set to 0
  - Owners with missing values → imputed using median (training fold only)

- **Rare category grouping** for selected categorical variables

- **Encoding strategies**
  - High-cardinality features → Target Encoding
  - Ordinal features → OrdinalEncoder with explicit ordering
  - Remaining categoricals → OneHotEncoder with infrequent category handling

- **Numeric transformations**
  - Log-transformation (`log1p`) for highly skewed financial variables
  - Median imputation for numeric variables
  - Outlier treatment tested for linear/distance-based models

This design ensures reproducibility and avoids data leakage during model selection.

---

## Model Comparison

The following model families were evaluated:

- Logistic Regression (Ridge, Lasso, Elastic Net)
- Linear SVC
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- LightGBM
- CatBoost
- HistGradientBoosting

Hyperparameter tuning was performed using `RandomizedSearchCV` with stratified cross-validation.  
Due to computational constraints, tuning was performed on a stratified training subsample while final evaluation was conducted on a held-out test set.

---

## Evaluation Metrics

Because of class imbalance, model comparison prioritizes:

- **PR-AUC (Average Precision)** — performance on the minority class  
- **ROC-AUC** — ranking performance across thresholds  

Additional diagnostics include:

- Precision / Recall / F1-score  
- Balanced Accuracy  
- Matthews Correlation Coefficient (MCC)  
- Confusion Matrix  
- ROC and Precision–Recall curves  

---

## Key Results

- Regularized logistic regression provides a stable and interpretable baseline.
- Linear SVC performs comparably to logistic regression.
- Distance-based methods (KNN) underperform due to high dimensionality and imbalance.
- Single decision trees exhibit weak generalization.
- Ensemble tree-based models significantly improve ranking performance.

**LightGBM achieved the strongest cross-validated PR-AUC and ROC-AUC and was selected as the final model.**

On the held-out test set:
- ROC-AUC ≈ 0.69  
- PR-AUC ≈ 0.29  
- MCC ≈ 0.20  

The model demonstrates moderate discriminative ability: it successfully ranks defaulters above non-defaulters better than random but still exhibits substantial overlap between the two groups.

---

## Threshold Analysis

Instead of relying on a default 0.50 classification threshold, the project explores operationally meaningful thresholds:

- Flag top 10% highest-risk applicants  
- Flag top 20% highest-risk applicants  
- Select threshold based on recall–precision trade-off  

Results show that increasing recall significantly increases false positives, highlighting the importance of cost-sensitive decision rules.

In practice, the model is better suited for:
- Risk ranking  
- Prioritized manual review  
- Portfolio monitoring  

rather than fully automated approval decisions.

---

## Limitations

- This is an analytical case study using public data.
- Model outputs are risk scores and may require calibration before interpretation as probabilities.
- Performance may degrade under distribution shift.
- Gradient boosting models reduce interpretability relative to linear models (feature importance / SHAP can help).

---

## Repository Structure
credit-default-prediction/
├── notebooks/
│ └── credit_default_modeling.ipynb
├── src/
│ ├── preprocessing.py
│ ├── modeling.py
│ └── utils.py
├── requirements.txt
└── README.md

