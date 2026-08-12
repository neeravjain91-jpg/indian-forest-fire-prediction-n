# Experimental protocol

## Target

One observation is a 0.1° latitude/longitude grid cell at a specific UTC hour and date. `fire=1` means at least one SNPP-VIIRS FIRMS detection was aggregated into that cell/time. `fire=0` is a sampled non-detection and is not treated as proof that no physical fire existed.

## Primary temporal protocol

- Train: 2018–2023
- Validation: 2024
- Final test: 2025

The 2025 test set is never used for feature selection, threshold selection, hyperparameter tuning, or model selection.

## Spatial protocol

Observations are assigned to 2° geographic blocks. Blocks, not individual rows, are randomly divided into train/test groups. This prevents neighboring grid cells from being scattered across both sets.

## Metrics

Because fire occurrence is a rare-event problem, the primary metrics are:

- PR-AUC
- ROC-AUC
- F1
- precision
- recall
- Brier score

Accuracy is not a primary metric.

## Ablation

The following feature sets are compared:

1. Instantaneous meteorology.
2. Instantaneous + 24 h summaries.
3. Instantaneous + 24/72/168 h rainfall history.
4. Full multiscale meteorology.

## Baselines

- Logistic Regression
- Random Forest
- Gradient Boosting

These establish whether the observed research effect is robust across conventional model families.

## Leakage controls

- No random row split is used for the primary future-year test.
- 2025 is completely held out.
- Spatial holdout operates at group level.
- The weather windows are historical windows ending at the target observation time; the paper must not describe this retrospective setup as an operational forecast unless a separate lagged/forecast-input experiment is implemented.

## Reproducibility

All sampling uses fixed seeds. Dataset-generation scripts preserve the year and grid identifiers needed to reproduce the splits.
