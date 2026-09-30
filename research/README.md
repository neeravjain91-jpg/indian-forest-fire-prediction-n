# Research Experimental Protocol & Reproducibility Track

This directory contains the reproducible research track for the India Forest Fire Occurrence Classification project. It is strictly separated from the operational live surveillance web interface.

## Core Research Question

How much predictive signal for VIIRS active-fire occurrence across India is driven by:
1. Antecedent and multi-timescale meteorological conditions (1-day, 3-day, 7-day ERA5-Land drying),
2. Geographic location and spatial clustering (grid latitude / longitude), and
3. Seasonal / temporal diurnal structure (month, hour, year)?

## Dataset & Protocol Specification

- **Dataset**: `data/processed/india_fire_weather_final.csv`
  - 131,000 observations (65,518 fire, 65,482 non-fire)
  - 26,494 unique 0.1° spatial grid cells across sovereign India
  - Chronological span: 2018–2025
  - Exactly 31 predictive features (2 spatial, 3 temporal, 26 multi-timescale weather)
- **Chronological Split Protocol**:
  - Training: 2018–2022 (84,661 observations)
  - Validation: 2023 (14,814 observations)
  - Final Held-Out Test: 2024–2025 (31,525 observations)
  - Strict leakage control: future years are completely held out from feature engineering, hyperparameter selection, and threshold tuning.

## Verified Results (Test Set 2024–2025)

| Metric | HistGradientBoosting (Primary) | Random Forest | Logistic Regression |
|---|---|---|---|
| **ROC-AUC** | **0.785174** | 0.584555 | 0.574444 |
| **PR-AUC** | **0.783202** | 0.569246 | 0.561531 |
| **F1-Score** | **0.712207** | 0.579234 | 0.561666 |
| **Accuracy** | **0.700111** | 0.553434 | 0.550738 |
| **Precision** | **0.684654** | 0.547643 | 0.548377 |
| **Recall** | **0.742071** | 0.614692 | 0.575615 |

Confusion Matrix (Test Set):
```
[[10373,  5388],
 [ 4066, 11698]]
```

## Reproducibility Scripts

- `research/model_training.py`: Retrains the primary `HistGradientBoostingClassifier` and outputs evaluation metrics.
- `research/evaluate_baselines_and_ablations.py`: Evaluates baselines (LR, RF, HGB) and all 6 ablation feature subsets into `results/baselines_and_ablations.json`.
- `tests/test_model_inference.py`: Automated pytest test asserting metrics consistency against `results/final_model/metrics.json`.

