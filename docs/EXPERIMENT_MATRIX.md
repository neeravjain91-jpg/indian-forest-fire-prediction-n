# Experiment Matrix & Scientific Findings

## 1. Baseline Model Comparison (Canonical 31 Features)

Evaluated under the fixed chronological protocol (Train: 2018–2022, $N=84,661$; Test: 2024–2025, $N=31,525$):

| Model Architecture | Features | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 31 | 55.07% | 54.84% | 57.56% | 56.17% | 57.44% | 56.15% |
| **Random Forest** | 31 | 55.34% | 54.76% | 61.47% | 57.92% | 58.46% | 56.92% |
| **HistGradientBoosting (Primary)** | **31** | **70.01%** | **68.47%** | **74.21%** | **71.22%** | **78.52%** | **78.32%** |

### Key Baseline Takeaway
Linear modeling (`Logistic Regression`) and unpruned ensemble bagging (`Random Forest`) achieve modest discrimination (~57–58% ROC-AUC). `HistGradientBoostingClassifier` effectively captures non-linear interactions across spatial coordinates and multi-timescale meteorological thresholds, achieving **70.01% Test Accuracy** and **78.52% ROC-AUC**.

---

## 2. Controlled Feature Ablation Matrix (HistGradientBoosting)

To systematically decouple the predictive contribution of spatial coordinates, temporal cycles, and antecedent meteorology, six feature subsets were evaluated under identical chronological splits:

| Ablation ID | Feature Group Description | Count | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full 31 Features** | All spatial, temporal, and multi-timescale weather features | 31 | 70.01% | 68.47% | 74.21% | 71.22% | 78.52% | 78.32% |
| **Coordinates + Temporal** | Spatial coordinates (`grid_lat`, `grid_lon`) + cyclical time (`hour`, `year`, `month`) | 5 | 69.79% | 66.56% | 79.55% | 72.48% | 80.70% | 81.37% |
| **Coordinates + Weather** | Spatial coordinates + 26 multi-timescale weather features (no temporal cyclical) | 28 | 69.34% | 67.33% | 75.12% | 71.01% | 77.37% | 77.58% |
| **Weather Only** | 1-day, 3-day, and 7-day meteorological features alone (no spatial coordinates) | 26 | 55.26% | 54.73% | 61.01% | 57.70% | 57.59% | 56.11% |
| **Coordinates Only** | Latitude and longitude spatial grid coordinates alone | 2 | 69.75% | 66.38% | 80.08% | 72.59% | 80.68% | 81.34% |
| **Temporal Only** | Observation hour, year, and month alone | 3 | 50.00% | 50.01% | 67.15% | 57.32% | 50.02% | 50.01% |

---

## 3. Scientific Findings & Interpretation

1. **Dominance of Spatial Location**:
   Spatial coordinates alone (`grid_lat`, `grid_lon`, 2 features) yield an ROC-AUC of **80.68%** and PR-AUC of **81.34%**. Wildfires in India are heavily geographically clustered in specific biomes and forest tracts (e.g., Central India, Western Ghats, Similipal, Northeast hills). Spatial location accounts for the vast majority of discriminative fire occurrence signal.

2. **Modest Signal from Pure Meteorology**:
   Multi-timescale weather features alone (26 features) achieve an ROC-AUC of **57.59%** and PR-AUC of **56.11%**. While extreme temperatures, low relative humidity, and dry antecedent conditions are necessary physical pre-conditions for fire ignition, hot and dry weather alone occurs over wide non-forested regions (e.g., arid northwest India) where biomass fuels are absent. Meteorology alone cannot distinguish fire occurrence without geographic spatial grounding.

3. **Temporal Cyclical Independence**:
   Temporal features alone (`hour`, `year`, `month`) produce near-chance discrimination (**50.02% ROC-AUC**), confirming that seasonal timing alone does not distinguish fire cells without spatial and climatic context.
