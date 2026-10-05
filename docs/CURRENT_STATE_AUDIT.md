# Repository State & Baseline System Audit

## 1. Executive Summary

This audit establishes the definitive baseline research state of the **India Forest Fire Occurrence Prediction** repository (`neeravjain91-jpg/indian-forest-fire-prediction-n`). 

The repository implements a compact, reproducible, research-oriented wildfire occurrence classification system across sovereign India (2018–2025). The central research question is:
> *How much predictive signal for wildfire occurrence across India comes from multi-timescale meteorological conditions, temporal/seasonal structure, and geographic location?*

All components of the project have been audited against the physical repository codebase, validated dataset, serialized models, test suite, and web application.

---

## 2. Component Inventory & Audit Matrix

| Component | Actual Implementation | Actual File Path | Verification Status | Final Status |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset** | 131,000 balanced observations across India (2018–2025) with 31 features and zero nulls | `data/processed/india_fire_weather_final.csv` | Validated via `tests/test_dataset_schema.py` (7 tests pass) | **Active / Core** |
| **National Boundary** | Official Survey of India sovereign boundary GeoJSON (193 KB) | `data/processed/india_boundary.geojson` | Validated via `tests/test_boundary_and_geometry.py` (4 tests pass) | **Active / Core** |
| **Primary Model** | `HistGradientBoostingClassifier` (31 features, `max_iter=300`, `lr=0.05`, `max_leaf_nodes=31`, `l2=1.0`) | `results/final_model/final_hgb_model.joblib` | Validated via `tests/test_model_inference.py` (7 tests pass) | **Active / Core** |
| **Model Metrics** | Validation and test set metrics, confusion matrix, and feature importances | `results/final_model/metrics.json` | Verified exact match against evaluated model outputs | **Active / Core** |
| **Test Predictions** | Complete test-set predictions (31,525 rows) with actual labels and probabilities | `results/final_model/test_predictions.csv` | Row count and calibration verified | **Active / Core** |
| **Feature Importance** | Permutation importance over test set (ROC-AUC drop over 5 repeats) | `results/final_model/feature_importance.csv` | Confirmed top 10 features | **Active / Core** |
| **Baseline Comparisons** | Logistic Regression, Random Forest, and HistGradientBoosting benchmarks | `results/baselines/baseline_comparison_metrics.csv` | Evaluated on identical temporal splits | **Active / Core** |
| **Feature Ablations** | 6 systematically evaluated feature subsets isolating spatial, temporal, and weather signals | `results/ablations/ablation_comparison.csv` | Full, coords+temp, coords+weather, weather-only, coords-only, temp-only | **Active / Core** |
| **ROC/PR Curves** | Visual validation curves for test set predictions | `results/figures/fig1_roc_pr_curves.png` | Saved high-resolution artifact | **Active / Core** |
| **Flask Application** | Dual-mode web interface: NASA FIRMS Live Observation + 31-Feature Occurrence Classifier | `application.py` | Validated via `tests/test_application_api.py` (8 tests pass) | **Active / Core** |
| **FIRMS Service** | Real-time / demo satellite ingestion with Survey of India spatial polygon filtering | `firms_service.py` | Validated via `tests/test_firms_service.py` (7 tests pass) | **Active / Core** |
| **Web UI** | GIS Leaflet interface with strict distinction between live observation and ML risk scoring | `templates/index.html` | Client-side map rendering and prediction form | **Active / Core** |
| **Training Pipeline** | Reproducible standalone training script | `train_final_model.py` | Validated end-to-end training and export | **Active / Core** |
| **Test Suite** | Comprehensive pytest suite covering schema, boundary, FIRMS, model, and application | `tests/` | 33 tests passing (100% pass rate) | **Active / Core** |

---

## 3. Strict Boundary & Separation from Major Research Track

In accordance with the project identity guidelines, this repository is strictly preserved as the **Mini / Baseline Research Project**. All speculative and advanced extensions belonging conceptually to the Major Project track have been eliminated from the repository:

1. **No Event-Centric Forecasting**: Eliminates DBSCAN-ST spatiotemporal event clustering, cluster kinetics, and multi-day complex tracking.
2. **No Multi-Horizon Lead Predictions ($T+24\text{h}$, $T+48\text{h}$)**: Occurrence target is strictly synchronous occurrence $Y \in \{0, 1\}$ at reference observation time $T$.
3. **No Multimodal Deep Learning**: No PyTorch dependencies, recurrent networks, or multimodal fusion encoders.
4. **No Historical Replay Engine**: No retrospective spatial replay engine or synthetic simulation apparatus.
5. **No Terrain/Geomorphological Augmentations**: The 31-feature space relies purely on meteorological reanalysis and coordinates, without external DEM raster dependencies.
6. **No LOEO / LOGRO Cross-Regional Protocols**: Evaluated via standard chronological train/val/test splits and 2-degree spatial block generalization.

---

## 4. Final Verification Summary

- **Automated Tests**: 33 passed, 0 failed (`python -m pytest -v`)
- **Execution Time**: ~4.0 seconds
- **Dependencies**: Minimal (`numpy`, `pandas`, `scikit-learn`, `requests`, `joblib`, `Flask`, `shapely`, `pytest`)
- **Model Checkpoint**: 1.15 MB serialized joblib model
- **Codebase Cleanliness**: Zero broken imports, zero machine-specific paths, zero committed secrets
