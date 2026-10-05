# Data Leakage Prevention Policy

## 1. Core Principles

Data leakage is the most pervasive failure mode in geospatial and meteorological machine learning, often leading to unrealistically optimistic reported performance that collapses upon prospective deployment. This repository enforces three absolute leakage prevention policies:

---

## 2. Policy 1: Temporal Directionality & Antecedent Gating

1. **Strictly Backward-Looking Feature Windows**:
   All 26 meteorological predictors are computed strictly over antecedent time horizons prior to or at reference observation time $T$:
   - 1-day features: Aggregated over $[T - 24\text{h}, T]$.
   - 3-day features: Aggregated over $[T - 72\text{h}, T]$.
   - 7-day features: Aggregated over $[T - 168\text{h}, T]$.
2. **Zero Forward Infiltration**: Under no circumstances do meteorological values from $t > T$ enter feature construction.
3. **Temporal Partitioning**: The training set ($T \le 2022$), validation set ($T = 2023$), and test set ($T \ge 2024$) are separated by strict calendar boundaries.

---

## 3. Policy 2: Satellite Target Telemetry Exclusion

A common flaw in remote-sensing fire models is accidentally including direct satellite sensor measurements of the fire thermal anomaly as predictive features.

The following variables are **strictly quarantined as non-predictive targets or metadata**:
- **Fire Radiative Power (`frp`)**: Represents instantaneous radiant heat emission from active combustion. Excluded.
- **Brightness Temperature (`brightness`, `bright_t31`)**: Direct thermal infrared brightness measurements. Excluded.
- **Algorithm Confidence Flag (`confidence`)**: Derived from thermal signature contrast against background. Excluded.
- **Sensor Telemetry (`scan`, `track`, `satellite`)**: Sensor orbit parameters. Excluded.

Only spatiotemporal coordinates and independent external ERA5-Land meteorology are permitted in the predictive feature vector $X$.

---

## 4. Policy 3: Preprocessing & Scaling Isolation

1. **No Target Leakage in Negative Sampling**: Negative (non-fire) samples are drawn only from valid spatial surveillance locations within matching temporal periods.
2. **Dataset-Level Independence**: Tree-based algorithms (such as `HistGradientBoostingClassifier`) operate via ordinal feature binning that does not rely on global test-set statistics.
3. **Immutability of Test Set**: The test split (2024–2025) is sealed and evaluated once without iterative test-set tuning.
