# Empirical Results & Model Performance

## 1. Primary Validated Model

The authoritative model for this repository is the 31-feature **HistGradientBoostingClassifier** trained on historical multi-timescale meteorology and spatial coordinates:
- **Model Checkpoint**: `results/final_model/final_hgb_model.joblib`
- **Configuration**:
  - `max_iter`: 300
  - `learning_rate`: 0.05
  - `max_leaf_nodes`: 31
  - `l2_regularization`: 1.0
  - `random_state`: 42

---

## 2. Chronological Test Performance (2024–2025 Held-Out Split)

Evaluated on $N = 31,525$ prospective test observations (15,762 fire detections, 15,763 non-fire controls) completely held out from model training and validation:

| Metric | Test Value (2024–2025) | Validation Value (2023) | Description |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **70.0111%** | 70.3389% | Overall classification correctness at threshold 0.50 |
| **Precision** | **68.4654%** | 68.8369% | Positive predictive value (true fires / predicted fires) |
| **Recall** | **74.2071%** | 74.2672% | Sensitivity / detection rate (true fires detected) |
| **F1-Score** | **71.2207%** | 71.4490% | Harmonic mean of precision and recall |
| **ROC-AUC** | **78.5174%** | 78.8008% | Area under receiver operating characteristic curve |
| **PR-AUC** | **78.3202%** | 78.2810% | Area under precision-recall curve |

### 2.1 Test Confusion Matrix

```
                      Predicted Non-Fire (0)    Predicted Fire (1)
Actual Non-Fire (0):         10,373 (TN)               5,388 (FP)
Actual Fire (1):              4,066 (FN)              11,698 (TP)
```
- **True Positives ($TP$)**: 11,698
- **True Negatives ($TN$)**: 10,373
- **False Positives ($FP$)**: 5,388
- **False Negatives ($FN$)**: 4,066
- **Test Observations**: 31,525

---

## 3. Feature Importance Analysis

Permutation feature importance was computed on the test set using ROC-AUC degradation over 5 repeated shuffles (`results/final_model/feature_importance.csv`):

| Rank | Feature Name | Category | Permutation Importance (ROC-AUC Drop) |
| :---: | :--- | :--- | :---: |
| 1 | `grid_lon` | Spatial Coordinate | **0.16131 ± 0.00030** |
| 2 | `grid_lat` | Spatial Coordinate | **0.06652 ± 0.00133** |
| 3 | `rh_1d` | 1-Day Weather | **0.06256 ± 0.00166** |
| 4 | `rain_1d` | 1-Day Weather | **0.01841 ± 0.00123** |
| 5 | `month` | Temporal Cyclical | **0.01188 ± 0.00071** |
| 6 | `rh_7d_mean` | 7-Day Antecedent Weather | **0.00864 ± 0.00094** |
| 7 | `rh_7d_min` | 7-Day Antecedent Weather | **0.00685 ± 0.00035** |
| 8 | `soil_7d_mean` | 7-Day Antecedent Weather | **0.00395 ± 0.00045** |
| 9 | `rain_7d_total` | 7-Day Antecedent Weather | **0.00296 ± 0.00049** |
| 10 | `soil_1d` | 1-Day Weather | **0.00286 ± 0.00032** |

> [!NOTE]
> Permutation importance reflects statistical predictive utility within this model architecture and dataset formulation; it should not be interpreted as direct physical causality.

---

## 4. Multi-Baseline Comparison

| Baseline Model | Feature Space | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 31 | 55.07% | 54.84% | 57.56% | 56.17% | 57.44% | 56.15% |
| Random Forest | 31 | 55.34% | 54.76% | 61.47% | 57.92% | 58.46% | 56.92% |
| **HistGradientBoosting (Final)** | **31** | **70.01%** | **68.47%** | **74.21%** | **71.22%** | **78.52%** | **78.32%** |

---

## 5. Summary of Controlled Ablations

| Feature Group | Count | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full 31 Features** | 31 | 70.01% | 68.47% | 74.21% | 71.22% | 78.52% | 78.32% |
| Coordinates + Temporal | 5 | 69.79% | 66.56% | 79.55% | 72.48% | 80.70% | 81.37% |
| Coordinates + Weather | 28 | 69.34% | 67.33% | 75.12% | 71.01% | 77.37% | 77.58% |
| Weather Only | 26 | 55.26% | 54.73% | 61.01% | 57.70% | 57.59% | 56.11% |
| Coordinates Only | 2 | 69.75% | 66.38% | 80.08% | 72.59% | 80.68% | 81.34% |
| Temporal Only | 3 | 50.00% | 50.01% | 67.15% | 57.32% | 50.02% | 50.01% |
