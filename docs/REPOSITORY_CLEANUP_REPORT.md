# Repository Cleanup & Mini/Major Separation Delivery Report

## 1. Executive Summary

This report documents the rigorous cleanup, verification, and complete separation of the **India Forest Fire Occurrence Prediction Mini / Baseline Research Project** (`neeravjain91-jpg/indian-forest-fire-prediction-n`) from the speculative architecture of the Major Research Project.

All Major-project contamination—including event-centric spatiotemporal clustering, forward multi-horizon forecasting ($T+24\text{h}$, $T+48\text{h}$), multimodal deep learning, DEM raster slope/ruggedness pipelines, regional LOEO/LOGRO protocols, and historical replay stations—has been systematically removed. The repository now stands as a clean, self-contained, reproducible, and verifiable baseline machine learning research system.

---

## 2. Removed Major Contamination

The following files and directories belonging exclusively to the Major research track were deleted from this repository:

### 2.1 Major Source Modules
- `src/` (Entirely removed):
  - `src/events/event_clustering.py` (DBSCAN-ST spatiotemporal event clustering)
  - `src/replay/historical_replay.py` (Historical replay simulation engine)
  - `src/models/multimodal_deep.py` (PyTorch BiGRU deep neural network)
  - `src/models/multi_horizon_eval.py` (Multi-horizon forward forecasting)
  - `src/models/ablation_study.py` (Major 39-feature multimodal ablations)
  - `src/data/terrain.py` (SRTM DEM slope and ruggedness calculations)
  - `src/data/environmental.py` (Ecological regime and VPD transformations)
  - `src/data/dataset_builder.py` (Major 39-feature dataset builder)
  - `src/evaluation/geographic_eval.py` (LOGRO regional holdout evaluation)
  - `src/evaluation/statistical_testing.py` (2x2 factorial bootstrap confidence intervals)
  - `src/evaluation/calibration.py` (Platt vs Isotonic calibration experiments)
  - `src/evaluation/generate_figures.py` (Major publication figure scripts)

### 2.2 Major-Only Results & Generated Artifacts
- `results/geographic/` (LOEO / LOGRO cross-regional evaluations)
- `results/multimodal/` (39-feature temporal multimodal predictions and quantile uncertainty)
- `results/multi_horizon/` (T+24h, T+48h forecasting comparisons)
- `results/replay/` (20-date historical replay benchmarks)
- `results/uncertainty/` (Epistemic and aleatoric uncertainty files)
- `results/baselines/bootstrap_confidence_intervals.csv`
- `results/baselines/calibration_comparison.csv`
- `results/baselines/factorial_interaction_analysis.csv`
- `results/baselines/precision_recall_at_k.csv`
- `results/figures/fig2_reliability_diagrams.png`
- `results/figures/fig3_bootstrap_confidence_intervals.png`
- `results/figures/fig4_loeo_geographic_spread.png`
- `data/processed/india_srtm_dem_01deg.csv` (SRTM digital elevation model)

### 2.3 Major-Specific Research Documents
- `docs/MAJOR_PROJECT_ARCHITECTURE.md`
- `docs/LITERATURE_GAP.md`
- `docs/RESEARCH_QUESTIONS.md`
- `tests/test_wildfire_research.py` (Major-specific test suite)

### 2.4 Application Routes Cleaned
- Removed endpoints from `application.py`:
  - `POST /api/forecast`
  - `GET /api/active-events`
  - `GET /api/historical-replay`
  - `GET /api/research-status`

---

## 3. Retained & Validated Mini Core Architecture

| Component | Path | Specification |
| :--- | :--- | :--- |
| **Canonical Dataset** | `data/processed/india_fire_weather_final.csv` | 131,000 balanced observations across India (2018–2025) with exactly 31 features and zero missing values. |
| **Spatial Boundary** | `data/processed/india_boundary.geojson` | Official Survey of India GeoJSON polygon for territorial integrity filtering. |
| **Final Trained Model** | `results/final_model/final_hgb_model.joblib` | Serialized `HistGradientBoostingClassifier` trained on 2018–2022. |
| **Evaluation Metrics** | `results/final_model/metrics.json` | Test Accuracy: 70.01%, ROC-AUC: 78.52%, PR-AUC: 78.32%, F1: 71.22%. |
| **Test Predictions** | `results/final_model/test_predictions.csv` | 31,525 rows of prospective predictions with actual labels and calibrated probabilities. |
| **Feature Importance** | `results/final_model/feature_importance.csv` | Permutation importance over 5 repeats. |
| **Baselines Benchmark** | `results/baselines/` | Comparative benchmark of Logistic Regression, Random Forest, and HistGradientBoosting. |
| **Ablation Benchmark** | `results/ablations/` | 6 controlled feature ablations isolating coordinates, temporal cycles, and weather windows. |
| **Training Pipeline** | `train_final_model.py` | Standalone, fully reproducible script to retrain the final HGB model from scratch. |
| **Web Application** | `application.py` | Flask service presenting Live Satellite Observation and 31-Feature Model inference. |
| **Satellite Service** | `firms_service.py` | Secure NASA FIRMS VIIRS 375m client with Survey of India spatial polygon enforcement. |
| **Web Interface** | `templates/index.html` | GIS dashboard cleanly separating live satellite detections from historical ML classification. |

---

## 4. Final Scientific Specifications

- **Feature Count**: Exactly **31 predictive features**
  - Spatial (2): `grid_lat`, `grid_lon`
  - Temporal (3): `hour`, `year`, `month`
  - 1-Day Weather (6): `temp_1d`, `rh_1d`, `wind_1d`, `pressure_1d`, `soil_1d`, `rain_1d`
  - 3-Day Weather (10): `temp_3d_mean/max/min`, `rh_3d_mean/min`, `wind_3d_mean/max`, `pressure_3d_mean`, `soil_3d_mean`, `rain_3d_total`
  - 7-Day Weather (10): `temp_7d_mean/max/min`, `rh_7d_mean/min`, `wind_7d_mean/max`, `pressure_7d_mean`, `soil_7d_mean`, `rain_7d_total`
- **Primary Target**: Binary fire occurrence $Y \in \{0, 1\}$ at reference observation time $T$.
- **Temporal Split**: Train 2018–2022 ($N = 84,661$), Val 2023 ($N = 14,814$), Test 2024–2025 ($N = 31,525$).
- **Primary Classifier**: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)`.

---

## 5. Verification & Test Suite Results

The entire automated test suite was executed via `pytest -v`:
- `tests/test_application_api.py`: 8 tests passed
- `tests/test_boundary_and_geometry.py`: 4 tests passed
- `tests/test_dataset_schema.py`: 7 tests passed
- `tests/test_firms_service.py`: 7 tests passed
- `tests/test_model_inference.py`: 7 tests passed
- **Total Test Count**: **33 tests**
- **Pass Rate**: **100% (33 / 33 passed)**
- **Test Runtime**: ~4.0 seconds

---

## 6. Known Limitations Disclosed

1. **Retrospective Case-Control Design**: The dataset is balanced 1:1 between fire and non-fire cells, meaning raw model output probabilities reflect sample odds rather than unconditional national prevalence.
2. **Satellite Detection Latency & Occlusion**: VIIRS observations represent satellite thermal anomalies at overpass times and can be occluded by heavy cloud or smoke.
3. **Observational vs. Causal Correlation**: Meteorological feature importance indicates predictive conditioning, not physical causal intervention effects.
