# Event-Centric Multimodal Spatiotemporal Wildfire Intelligence for India

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20LightGBM-orange.svg)](https://pytorch.org/)
[![Evaluation: Disjoint Holdouts](https://img.shields.io/badge/Evaluation-Spatially%20Disjoint-brightgreen.svg)](docs/EVALUATION_PROTOCOL.md)

An event-centric, multimodal spatiotemporal wildfire forecasting and intelligence framework for sovereign India (2018–2025). Integrates **NASA FIRMS VIIRS 375m** satellite telemetry, **Copernicus ERA5-Land** multi-timescale atmospheric reanalysis, **NASA SRTM** terrain geomorphology, atmospheric fuel dryness dynamics, and connected-component spatiotemporal fire event tracking.

---

## 1. Major Research Question & Hypotheses

### Central Research Question
> *Can an event-centric spatiotemporal framework—combining multi-timescale atmospheric drying history, antecedent fire persistence, terrain geomorphology, and environmental fuel dryness—produce superior and more transferable forward wildfire forecasts ($T+24\text{h}$, $T+48\text{h}$) across India than static, location-dependent occurrence classifiers?*

### Formal Hypotheses & Verified Outcomes
* **H1 (Multi-timescale weather dynamics)**: **CONFIRMED**. Integrating antecedent multi-day drying signals (1d, 3d, 7d) improved test ROC-AUC from **56.25% to 57.84%** ($\Delta = +1.59\%$, $p < 0.01$) over instantaneous weather alone.
* **H2 (Event-centric representation)**: **CONFIRMED**. Incorporating active cluster proximity, duration, and antecedent recurrence boosted ROC-AUC to **58.94%** and achieved **62.30% ROC-AUC** on active multi-day complex persistence forecasting.
* **H3 (Multimodal geographic transferability)**: **CONFIRMED**. The 39-feature multimodal model outperformed the mini-project baseline by **+1.07% ROC-AUC** with superior probability calibration (ECE 0.0174 vs. 0.0211) when evaluated on an unseen held-out ecological regime (Central India Deciduous Forest).
* **H4 (Probability calibration & uncertainty)**: **CONFIRMED**. Isotonic calibration reduced Expected Calibration Error down to $0.012 - 0.014$, and epistemic uncertainty correlated monotonically with empirical prediction error.

---

## 2. Research Architecture & System Overview

```
                                  DATA SOURCES
   ┌────────────────────────┬────────────────────────┬────────────────────────┐
   │  NASA FIRMS VIIRS 375m │  Copernicus ERA5-Land  │   NASA SRTM 90m DEM    │
   │  (SNPP, NOAA20, NOAA21)│  (Hourly Reanalysis)   │   (Topography / TRI)   │
   └───────────┬────────────┴───────────┬────────────┴───────────┬────────────┘
               │                        │                        │
               ▼                        ▼                        ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                     SPATIOTEMPORAL NORMALIZATION GRID                      │
 │                     0.1° x 0.1° Regular Analysis Grid (~11 km)             │
 └──────────────────────────────────────┬─────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         ▼                                                             ▼
 ┌──────────────────────────────┐              ┌──────────────────────────────┐
 │   EVENT-CENTRIC TRACKING     │              │    MULTIMODAL FEATURE SPACE  │
 │ - Spatiotemporal Clustering  │              │ - 1d, 3d, 7d Weather Windows │
 │ - Connected Components       │              │ - Vapor Pressure Deficit     │
 │ - Centroid Trajectories      │              │ - Soil Drought Index         │
 │ - Event Duration & Dynamics  │              │ - Terrain Geomorphology      │
 └──────────────┬───────────────┘              │ - Antecedent Fire History    │
                │                              └──────────────┬───────────────┘
                └───────────────────────┬─────────────────────┘
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                     STRICT CAUSAL LEAKAGE CONTROL                          │
 │ - Features strictly at or prior to origin T                                │
 │ - Forward Targets: T (Diagnostic), T+24h (Next-Day), T+48h, Persistence    │
 └──────────────────────────────────────┬─────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         ▼                                                             ▼
 ┌──────────────────────────────┐              ┌──────────────────────────────┐
 │    BENCHMARK BASELINE SUITE  │              │   MULTIMODAL DEEP NETWORK    │
 │ - Logistic Regression        │              │ - Weather Encoder Branch     │
 │ - Random Forest              │              │ - Environment/Terrain Branch │
 │ - HistGradientBoosting (Mini)│              │ - Fire History Branch        │
 │ - LightGBM (Primary Tree)    │              │ - Gated Cross-Modality Fusion│
 │ - Multi-Layer Perceptron     │              │ - Multi-Task Prediction Heads│
 └──────────────┬───────────────┘              │ - MC Dropout Epistemic Std   │
                │                              └──────────────┬───────────────┘
                └───────────────────────┬─────────────────────┘
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                 RIGOROUS DUAL GENERALIZATION BENCHMARKS                    │
 │ 1. Chronological Test (Train: 2018-2022, Val: 2023, Test: 2024-2025)       │
 │ 2. Spatially Disjoint Holdout (Train: Non-Central Biomes, Test: Central)   │
 │ 3. Modality Ablation Matrix & Probability Calibration (ECE, Brier)         │
 └──────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                  INTELLIGENCE PLATFORM & HISTORICAL REPLAY                 │
 │ - Real-time NASA FIRMS India Live Surveillance                             │
 │ - Calibrated Multi-Horizon Forward Risk Inference                          │
 │ - Prospective Historical Replay & Spatial Hit/Miss Verification            │
 └────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset & Multimodal Features

* **Dataset Size**: 131,000 observations across sovereign India (2018–2025).
* **Spatial Resolution**: Regular $0.1^\circ \times 0.1^\circ$ grid cells ($\approx 11 \times 11\text{ km}$).
* **Temporal Coverage**: 8 full calendar years (2,662 active fire detection dates).
* **39 Total Engineered Predictors**:
  - **Atmospheric Weather History (26)**: 1d, 3d, 7d temperature, relative humidity, wind speed, surface pressure, soil moisture, precipitation.
  - **Terrain Geomorphology (3)**: Mean elevation, slope gradient, topographic ruggedness index (TRI).
  - **Environmental Fuel Dryness (3)**: 1d and 3d Vapor Pressure Deficit (VPD in kPa), topsoil moisture drought index.
  - **Antecedent Fire History (2)**: 2018–2022 cell-level recurrence rate, antecedent 24h fire indicator.
  - **Spatiotemporal Coordinates (5)**: Grid latitude, longitude, UTC hour, year, month.

---

## 4. Key Experimental Results

### Chronological Generalization (Test Set: 2024–2025)
*Trained on 2018–2022 ($N = 84,661$), Calibrated on 2023 ($N = 14,814$), Tested on 2024–2025 ($N = 31,525$):*

| Model Architecture | Feature Space | Calibrated Acc (%) | Calibrated ROC-AUC (%) | Calibrated PR-AUC (%) | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM (Major Project)** | 39 Multimodal | **57.12%** | **60.47%** | **58.04%** | **0.2400** | 0.0136 |
| **HistGradientBoosting (Mini Baseline)** | 39 Multimodal | 57.17% | 60.25% | 57.53% | 0.2403 | **0.0120** |
| **Multimodal Deep Net** | Modality Tensors | 56.52% | 59.59% | 57.18% | 0.2417 | 0.0160 |
| **Random Forest** | 39 Multimodal | 56.38% | 59.52% | 56.99% | 0.2417 | 0.0220 |
| **Logistic Regression** | 39 Multimodal | 55.92% | 58.71% | 56.35% | 0.2436 | 0.0269 |
| **MLP Baseline** | 39 Multimodal | 55.66% | 58.16% | 56.11% | 0.2451 | 0.0208 |

### Spatially Disjoint Regional Holdout (Central India Test)
*Trained on Non-Central biomes ($N = 83,603$), Tested on held-out Central India ($N = 47,397$):*
- **LightGBM Multimodal (39 feat)**: **59.41% ROC-AUC**, 56.58% PR-AUC, 64.34% F1, ECE **0.0174**.
- **HistGradientBoosting Mini Baseline (31 feat)**: 58.34% ROC-AUC, 55.95% PR-AUC, 62.90% F1, ECE 0.0211.
- **Multimodal Advantage**: **+1.07% ROC-AUC improvement** on unseen geographic domains.

---

## 5. Spatiotemporal Fire Event Clustering

By applying spatiotemporal connected-component clustering ($\epsilon_s = 25\text{ km}$, $\epsilon_t = 2\text{ days}$), the framework grouped 65,518 raw detections into **45,933 unique spatiotemporal fire events**.
- Mean duration: 1.03 days (maximum: 14 days).
- Multi-day fire complexes accounted for empirical centroid displacements averaging **16.34 km**.
- Forward event persistence forecasting achieved **62.30% ROC-AUC** on held-out test data.

---

## 6. Historical Replay & Spatial Verification

The platform includes a prospective **Historical Replay Engine** ([`src/replay/historical_replay.py`](file:///c:/Users/ASUS/Downloads/forest%20fire/src/replay/historical_replay.py)) that:
1. Gathers features strictly available at time $T$.
2. Computes the forward probability surface for $T+24\text{h}$.
3. Matches predictions against ground-truth satellite active fire observations at $T+24\text{h}$.
4. Computes spatial Hits, False Alarms, Misses, and Active Complex evolution.

---

## 7. Installation & Reproducibility

### 1. Setup Environment
```bash
git clone https://github.com/neeravjain91-jpg/major-forest-fire.git
cd major-forest-fire
pip install -r requirements.txt
```

### 2. Reproduce the Research Pipeline
```bash
# 1. Build multimodal features, event clusters, and rigorous splits
python -m src.data.dataset_builder

# 2. Train and benchmark the baseline suite
python -m src.models.baselines

# 3. Train the multimodal spatiotemporal deep learning model
python -m src.models.multimodal_deep --epochs 20

# 4. Run spatially disjoint geographic generalization benchmark
python -m src.evaluation.geographic_eval

# 5. Evaluate multi-horizon forward forecasting (T+24h, T+48h, Persistence)
python -m src.models.multi_horizon_eval

# 6. Execute controlled modality ablation study
python -m src.models.ablation_study

# 7. Generate publication-ready figures
python -m src.evaluation.generate_figures

# 8. Run scientific verification test suite
pytest -v tests/test_wildfire_research.py
```

### 3. Launch the Intelligence Platform
```bash
python application.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to access the Leaflet GIS interface.

---

## 8. Documentation Index

- [`docs/CURRENT_STATE_AUDIT.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/CURRENT_STATE_AUDIT.md): Detailed audit and migration record from mini-project baseline.
- [`docs/MAJOR_PROJECT_ARCHITECTURE.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/MAJOR_PROJECT_ARCHITECTURE.md): Research system architecture and data flows.
- [`docs/LITERATURE_GAP.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/LITERATURE_GAP.md): Survey of prior literature through 2026 and gap analysis.
- [`reports/literature_matrix.csv`](file:///c:/Users/ASUS/Downloads/forest%20fire/reports/literature_matrix.csv): Literature benchmarking matrix.
- [`docs/RESEARCH_QUESTIONS.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/RESEARCH_QUESTIONS.md): Formal research questions and falsification criteria.
- [`docs/DATA_SOURCES.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/DATA_SOURCES.md): Authoritative remote sensing and reanalysis documentation.
- [`docs/DATA_SCHEMA.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/DATA_SCHEMA.md): Feature dictionary and target definitions.
- [`docs/LEAKAGE_POLICY.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/LEAKAGE_POLICY.md): Strict causal gating and spatial autocorrelation policy.
- [`docs/EVALUATION_PROTOCOL.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/EVALUATION_PROTOCOL.md): Mathematical formulations of metrics and splits.
- [`docs/EXPERIMENT_MATRIX.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/EXPERIMENT_MATRIX.md): Reproducible experiment registry.
- [`docs/RESULTS.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/RESULTS.md): Detailed tables of all experimental outcomes.
- [`docs/DEPLOYMENT.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/DEPLOYMENT.md): Production deployment and security hardening guide.

---

## 9. Scientific Integrity & Limitations

1. **Defensible Forecast Horizons**: Because VIIRS active fire observations over India are acquired via sun-synchronous low Earth orbit passes (~2 passes per 24 hours per sensor), continuous hourly predictions ($T+1\text{h}$) are scientifically unsupportable without geostationary sensors. Defensible horizons are formulated at **$T+24\text{h}$** and **$T+48\text{h}$**.
2. **Empirical Kinematics vs. Physical Spread**: The system tracks statistical centroid displacement and cluster expansion; it does not solve Navier-Stokes physical fire spread models.
3. **Research Interface**: The web platform is a scientific exploration benchmark, **not an operational emergency warning system**.
