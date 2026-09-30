# Experimental protocol

## Target

One observation is a 0.1° latitude/longitude grid cell at a specific UTC hour and date. `fire=1` means at least one SNPP-VIIRS FIRMS detection was aggregated into that cell/time. `fire=0` is a sampled non-detection and is not treated as proof that no physical fire existed.

## Primary temporal protocol

- Train: 2018–2022 (84,661 observations)
- Validation: 2023 (14,814 observations)
- Final Test: 2024–2025 (31,525 observations)
- Total: 131,000 observations (65,518 fire, 65,482 non-fire)

The 2024–2025 held-out test set is never used for feature engineering, threshold tuning, or model selection.

## Spatial protocol

Observations are referenced to 0.1° grid cells (~11 km) across sovereign India (26,494 unique spatial cells). Spatial generalization is evaluated across geographic regions and holdout blocks to prevent spatial autocorrelation leakage.

## Model and Feature Specification

31 predictive features spanning multi-timescale atmospheric and spatiotemporal dimensions:
- Spatial (2): `grid_lat`, `grid_lon`
- Temporal (3): `hour`, `year`, `month`
- 1-day Weather (6): `temp_1d`, `rh_1d`, `wind_1d`, `pressure_1d`, `soil_1d`, `rain_1d`
- 3-day Weather (10): `temp_3d_mean`, `temp_3d_max`, `temp_3d_min`, `rh_3d_mean`, `rh_3d_min`, `wind_3d_mean`, `wind_3d_max`, `pressure_3d_mean`, `soil_3d_mean`, `rain_3d_total`
- 7-day Weather (10): `temp_7d_mean`, `temp_7d_max`, `temp_7d_min`, `rh_7d_mean`, `rh_7d_min`, `wind_7d_mean`, `wind_7d_max`, `pressure_7d_mean`, `soil_7d_mean`, `rain_7d_total`

Primary Classifier:
- `HistGradientBoostingClassifier` (max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)

## Metrics

Because fire occurrence is an imbalanced spatial classification problem, comprehensive discrimination and ranking metrics are reported:
- ROC-AUC
- PR-AUC
- F1-Score
- Precision
- Recall
- Accuracy
- Confusion Matrix

## Baselines & Ablations

1. Model Comparison:
   - Logistic Regression (with StandardScaler)
   - Random Forest
   - HistGradientBoostingClassifier (Primary Model)

2. Spatiotemporal & Meteorological Ablations:
   - Full 31 features
   - Coordinates + Temporal (5 features)
   - Coordinates + Weather (28 features)
   - Weather Only (26 features)
   - Coordinates Only (2 features)
   - Temporal Only (3 features)


## Leakage controls

- No random row split is used for the primary future-year test.
- 2025 is completely held out.
- Spatial holdout operates at group level.
- The weather windows are historical windows ending at the target observation time; the paper must not describe this retrospective setup as an operational forecast unless a separate lagged/forecast-input experiment is implemented.

## Reproducibility

All sampling uses fixed seeds. Dataset-generation scripts preserve the year and grid identifiers needed to reproduce the splits.
