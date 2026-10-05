# India Forest Fire Occurrence Prediction using Multi-Timescale Meteorological Features

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask%203.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-Passing%20(33%2F33)-brightgreen.svg)](tests/)

A reproducible machine learning research system evaluating wildfire occurrence classification across sovereign India (2018–2025) using **NASA FIRMS VIIRS (375m)** active-fire detections, **Copernicus ERA5-Land** multi-timescale atmospheric reanalysis, and an authoritative **HistGradientBoostingClassifier** baseline.

---

## 1. Problem Formulation

Wildfires in India pose significant threats to biodiversity, local livelihoods, and carbon storage across deciduous forests, tropical evergreen belts, and subtropical woodlands. Traditional satellite monitoring detects active fires at sensor overpass times, but remote sensing alone does not explain the antecedent meteorological drivers that make specific spatial cells susceptible to ignition. Understanding the interplay between instantaneous weather, multi-day fuel drying cycles, seasonal timing, and geographic location is critical for empirical fire risk assessment.

## 2. Research Objective

This research investigates the central empirical question:
> **How much predictive signal for wildfire occurrence across India comes from multi-timescale meteorological conditions, temporal/seasonal structure, and geographic location?**

The system quantifies:
1. The classification accuracy of tree-based gradient boosting on a controlled 1:1 case-control dataset.
2. The relative predictive utility of antecedent 1-day, 3-day, and 7-day atmospheric drying windows.
3. The degree to which spatial coordinates versus pure meteorological conditions drive occurrence predictions under strict chronological holdouts.

## 3. Data Sources

The project integrates three authoritative Earth observation and geographic datasets across sovereign India from 2018 to 2025:
1. **NASA FIRMS VIIRS Active Fire Telemetry**:
   - Sensor: Suomi-NPP VIIRS 375m I-Band (VNP14IMGTDL product).
   - Records: Active thermal anomalies across India from 2018 to 2025.
2. **Copernicus ERA5-Land Surface Meteorological Reanalysis**:
   - Provider: European Centre for Medium-Range Weather Forecasts (ECMWF).
   - Resolution: Hourly gridded surface variables at 0.10° spatial resolution (~9 km).
   - Parameters: 2m temperature, relative humidity, 10m wind speed, surface barometric pressure, topsoil moisture (0–7 cm layer 1), and total precipitation.
3. **Survey of India Sovereign Boundary**:
   - Official national administrative boundary GeoJSON (`data/processed/india_boundary.geojson`) ensuring all observations and live surveillance lie strictly within Indian territory.

## 4. Dataset Construction

Continuous satellite detections and gridded hourly weather series are synthesized into a canonical processed dataset:
`data/processed/india_fire_weather_final.csv`

- **Total Observations**: 131,000
- **Class Balance**: 65,518 fire detections ($Y=1$) vs. 65,482 matched non-fire controls ($Y=0$)
- **Spatial Resolution**: Uniform 0.10° latitude-longitude grid cells (~11.1 km)
- **Unique Spatial Cells**: 26,494 distinct geographic cells
- **Temporal Span**: January 1, 2018 to December 31, 2025 (8 full calendar years)
- **Data Integrity**: Exactly zero missing or null values across all 31 predictive features

## 5. Canonical 31-Feature Baseline

The feature vector contains exactly 31 predictive variables:
- **Spatial Coordinates (2)**: `grid_lat`, `grid_lon`
- **Temporal Cyclical (3)**: `hour` (UTC acquisition hour), `year`, `month`
- **1-Day Weather Window (6)**: `temp_1d`, `rh_1d`, `wind_1d`, `pressure_1d`, `soil_1d`, `rain_1d`
- **3-Day Antecedent Weather Window (10)**: `temp_3d_mean`, `temp_3d_max`, `temp_3d_min`, `rh_3d_mean`, `rh_3d_min`, `wind_3d_mean`, `wind_3d_max`, `pressure_3d_mean`, `soil_3d_mean`, `rain_3d_total`
- **7-Day Antecedent Weather Window (10)**: `temp_7d_mean`, `temp_7d_max`, `temp_7d_min`, `rh_7d_mean`, `rh_7d_min`, `wind_7d_mean`, `wind_7d_max`, `pressure_7d_mean`, `soil_7d_mean`, `rain_7d_total`

Direct satellite fire measurements (such as Fire Radiative Power, brightness temperature, and sensor confidence flags) are strictly excluded from the feature space to prevent data leakage.

## 6. Primary Machine Learning Model

The primary validated classifier is a `HistGradientBoostingClassifier` implemented via scikit-learn:
- **Max Iterations (`max_iter`)**: 300
- **Learning Rate (`learning_rate`)**: 0.05
- **Max Leaf Nodes (`max_leaf_nodes`)**: 31
- **L2 Regularization (`l2_regularization`)**: 1.0
- **Random Seed (`random_state`)**: 42
- **Model Checkpoint**: `results/final_model/final_hgb_model.joblib` (1.15 MB)

## 7. Temporal Evaluation Protocol

To prevent future lookahead leakage and evaluate true prospective generalization, data are strictly partitioned chronologically:
- **Training Set (2018–2022)**: 84,661 samples (64.6%)
- **Validation Set (2023)**: 14,814 samples (11.3%)
- **Prospective Test Set (2024–2025)**: 31,525 samples (24.1%)

All preprocessing, model fitting, and hyperparameter selections are finalized prior to evaluating the held-out 2024–2025 test partition.

## 8. Spatial Generalization & Feature Ablation Protocol

In addition to chronological evaluation, the study evaluates spatial generalization across held-out 2-degree geographic blocks and performs 6 controlled feature ablations to isolate the independent contribution of coordinates, temporal cycles, and antecedent weather windows.

## 9. Empirical Results

### 9.1 Test Set Performance (2024–2025 Held-Out Split)
Evaluated on $N = 31,525$ prospective test observations:

| Metric | Test Value (2024–2025) | Validation Value (2023) |
| :--- | :---: | :---: |
| **Accuracy** | **70.0111%** | 70.3389% |
| **Precision** | **68.4654%** | 68.8369% |
| **Recall** | **74.2071%** | 74.2672% |
| **F1-Score** | **71.2207%** | 71.4490% |
| **ROC-AUC** | **78.5174%** | 78.8008% |
| **PR-AUC** | **78.3202%** | 78.2810% |

#### Test Confusion Matrix ($N = 31,525$)
```
                      Predicted Non-Fire (0)    Predicted Fire (1)
Actual Non-Fire (0):         10,373 (TN)               5,388 (FP)
Actual Fire (1):              4,066 (FN)              11,698 (TP)
```

### 9.2 Baseline Model Comparison

| Model Architecture | Features | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 31 | 55.07% | 54.84% | 57.56% | 56.17% | 57.44% | 56.15% |
| Random Forest | 31 | 55.34% | 54.76% | 61.47% | 57.92% | 58.46% | 56.92% |
| **HistGradientBoosting (Final)** | **31** | **70.01%** | **68.47%** | **74.21%** | **71.22%** | **78.52%** | **78.32%** |

### 9.3 Controlled Feature Ablation Analysis

| Feature Group | Features | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full 31 Features** | 31 | 70.01% | 68.47% | 74.21% | 71.22% | 78.52% | 78.32% |
| Coordinates + Temporal | 5 | 69.79% | 66.56% | 79.55% | 72.48% | 80.70% | 81.37% |
| Coordinates + Weather | 28 | 69.34% | 67.33% | 75.12% | 71.01% | 77.37% | 77.58% |
| Weather Only | 26 | 55.26% | 54.73% | 61.01% | 57.70% | 57.59% | 56.11% |
| Coordinates Only | 2 | 69.75% | 66.38% | 80.08% | 72.59% | 80.68% | 81.34% |
| Temporal Only | 3 | 50.00% | 50.01% | 67.15% | 57.32% | 50.02% | 50.01% |

### 9.4 Key Scientific Insights
- **Spatial Coordinates Drive Occurrence Signal**: Spatial coordinates alone achieve **80.68% ROC-AUC**, reflecting the pronounced geographic clustering of wildfires in specific Indian forest tracts.
- **Meteorology Provides Physical Conditioning**: Weather features alone yield **57.59% ROC-AUC**, indicating that while atmospheric dryness is a necessary physical condition, it is insufficient on its own to predict fire occurrence without geographic fuel context.
- **Top Permutation Features**: `grid_lon`, `grid_lat`, `rh_1d`, `rain_1d`, `month`, `rh_7d_mean`, `rh_7d_min`, `soil_7d_mean`, `rain_7d_total`, and `soil_1d`.

## 10. Methodological Limitations

1. **Case-Control Sampling Design**: The dataset is balanced 1:1 ($P(Y=1) = 0.50$). Predicted probabilities represent conditional sample odds, not unconditional national population risk.
2. **Satellite Observation Constraints**: Polar-orbiting VIIRS satellites observe India roughly twice daily. A negative label ($Y=0$) indicates absence of satellite thermal detection during an overpass, which may be affected by orbital timing, cloud cover, or heavy smoke.
3. **Observational Correlation**: Permutation importance indicates statistical predictive utility rather than physical causal intervention effects.

## 11. Interactive Web Application

The repository includes a lightweight Flask application (`application.py`) providing two distinct interfaces:
1. **Live Satellite Observation**: Real-time NASA FIRMS VIIRS 375m active fire telemetry strictly masked to sovereign Indian territory via the Survey of India GeoJSON polygon.
2. **Historical Fire-Occurrence Classifier**: Direct interactive inference on the validated 31-feature model.

> **Operational Boundary Notice**: This application is an academic research demonstration for retrospective fire occurrence classification. Live FIRMS detections represent direct satellite sensor observations, not predictive model forecasts. The system does not issue emergency warnings.

## 12. Installation & Reproducibility

### 12.1 Environment Setup
```bash
git clone https://github.com/neeravjain91-jpg/indian-forest-fire-prediction-n.git
cd indian-forest-fire-prediction-n

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 12.2 Run Automated Verification Tests
```bash
python -m pytest -v
```
All 33 tests across dataset schema, spatial boundary geometry, FIRMS service, model inference, and web API endpoints should pass in ~4 seconds.

### 12.3 Retrain Model Pipeline from Scratch
```bash
python train_final_model.py --data data/processed/india_fire_weather_final.csv --output results/final_model
```

### 12.4 Launch Web Application
```bash
python application.py
```
Open `http://127.0.0.1:5000` in any web browser.
