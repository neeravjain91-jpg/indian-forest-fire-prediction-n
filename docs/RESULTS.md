# Experimental Results & Rigorous Hypothesis Verification

All results presented herein are derived from real, verified observations from the 131,000-observation India wildfire dataset (2018–2025) using real NOAA ETOPO/SRTM digital elevation data, strictly causal time-indexed fire history ($t < T$), and connected-component spatiotemporal event persistence targets.

---

## 1. Controlled 2x2 Factorial Baseline Comparison (Strict Chronological Test 2024–2025)

*Training Set*: 2018–2022 ($N = 84,661$) | *Validation Set*: 2023 ($N = 14,814$) | *Test Set*: 2024–2025 ($N = 31,525$).
*All models evaluated under identical splits and post-hoc isotonic probability calibration:*

| Experiment ID | Model Architecture | Feature Space | Calibrated Acc (%) | Calibrated F1 (%) | Calibrated ROC-AUC (%) | Calibrated PR-AUC (%) | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp A** | HistGradientBoosting (Mini Baseline) | 31 Baseline | 55.85% | 63.47% | 59.03% | 56.74% | 0.2427 | 0.0137 |
| **Exp B** | HistGradientBoosting | 39 Multimodal | 57.85% | 59.52% | 61.73% | 59.68% | 0.2384 | 0.0177 |
| **Exp C** | LightGBM | 31 Baseline | 56.33% | 62.47% | 59.23% | 57.00% | 0.2424 | 0.0153 |
| **Exp D** | **LightGBM (Primary Major)** | **39 Multimodal** | **58.44%** | 59.03% | **62.18%** | **60.15%** | **0.2375** | 0.0171 |
| *Ref* | Random Forest | 39 Multimodal | 57.51% | 61.83% | 60.94% | 58.51% | 0.2402 | 0.0172 |
| *Ref* | Logistic Regression | 39 Multimodal | 55.97% | 58.21% | 58.40% | 56.18% | 0.2446 | 0.0225 |
| *Ref* | Spatiotemporal BiGRU Deep Net | Modality Tensors | 55.91% | 57.14% | 58.92% | 57.61% | 0.2434 | 0.0190 |

---

## 2. Statistical Testing: Non-Parametric Bootstrap 95% Confidence Intervals

*Computed over $B = 1,000$ bootstrap resamples of the unseen test set ($N = 31,525$):*

| Comparison | Metric | Observed Difference ($\Delta$) | 95% Bootstrap Confidence Interval | Excludes Zero? | Statistical Inference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multimodal Effect in HGB** (Exp B vs Exp A) | ROC-AUC | **+0.0271 (+2.71%)** | **[+0.0216, +0.0322]** | **True** | Statistically distinguishable gain |
| **Multimodal Effect in HGB** (Exp B vs Exp A) | PR-AUC | **+0.0294 (+2.94%)** | **[+0.0240, +0.0352]** | **True** | Statistically distinguishable gain |
| **Multimodal Effect in HGB** (Exp B vs Exp A) | Brier Score | **-0.0044** | **[-0.0053, -0.0034]** | **True** | Statistically distinguishable error reduction |
| **Multimodal Effect in LGBM** (Exp D vs Exp C) | ROC-AUC | **+0.0295 (+2.95%)** | **[+0.0242, +0.0344]** | **True** | Statistically distinguishable gain |
| **Multimodal Effect in LGBM** (Exp D vs Exp C) | PR-AUC | **+0.0315 (+3.15%)** | **[+0.0260, +0.0371]** | **True** | Statistically distinguishable gain |
| **Multimodal Effect in LGBM** (Exp D vs Exp C) | Brier Score | **-0.0050** | **[-0.0059, -0.0040]** | **True** | Statistically distinguishable error reduction |
| **Model Family in 31 Baseline** (Exp C vs Exp A) | ROC-AUC | +0.0021 (+0.21%) | [-0.0014, +0.0053] | **False** | *Not statistically distinguishable* |
| **Model Family in 31 Baseline** (Exp C vs Exp A) | PR-AUC | +0.0026 (+0.26%) | [-0.0009, +0.0060] | **False** | *Not statistically distinguishable* |
| **Model Family in 39 Multimodal** (Exp D vs Exp B) | ROC-AUC | +0.0045 (+0.45%) | [+0.0018, +0.0072] | **True** | Statistically distinguishable gain |

> **Key Scientific Takeaway**: On baseline features alone, LightGBM is not statistically distinguishable from HistGradientBoosting ($95\%\text{ CI}$ crosses zero). The substantial $+2.71\%$ to $+2.95\%$ boost in ROC-AUC is driven specifically by the **multimodal feature expansion** (real DEM terrain, atmospheric fuel dryness, and causal fire history).

---

## 3. Leave-One-Ecoregion-Out (LOEO) Geographic Cross-Validation

*Trained on 5 biomes, tested exclusively on the 6th held-out biome across India:*

| Held-Out Ecoregion | HGB (31 Baseline) ROC-AUC (%) | LightGBM (39 Multimodal) ROC-AUC (%) | $\Delta$ ROC-AUC (%) | HGB Brier Score | LGBM Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CENTRAL** (Deccan dry deciduous) | 58.34% | **65.02%** | **+6.68%** | 0.2442 | **0.2331** |
| **WESTERN_GHATS** (Moist evergreen/montane) | 55.65% | **63.90%** | **+8.25%** | 0.2482 | **0.2343** |
| **NORTHEAST** (Subtropical Purvanchal) | 60.70% | **65.97%** | **+5.27%** | 0.2420 | **0.2354** |
| **NORTH** (Himalayan pine/oak) | 58.11% | **64.16%** | **+6.05%** | 0.2448 | **0.2366** |
| **EAST** (Eastern Ghats/Chota Nagpur) | 58.25% | **63.34%** | **+5.09%** | 0.2449 | **0.2367** |
| **NORTHWEST** (Semi-arid thorn scrub) | 55.45% | **65.07%** | **+9.62%** | 0.2483 | **0.2318** |
| **Macro Cross-Regional Mean** | **57.75% ± 1.96%** | **64.58% ± 0.95%** | **+6.83%** | **0.2454** | **0.2346** |

> **Key Finding**: LightGBM with 39 multimodal features outperforms the 31-feature baseline across all six held-out ecoregions. Furthermore, the cross-regional variance is reduced from $\pm 1.96\%$ down to $\pm 0.95\%$, demonstrating that real DEM terrain and environmental covariates stabilize out-of-region generalizability.

---

## 4. Multi-Horizon Forecasting & Rare-Event Precision@k

| Horizon | Target Formulation | Positive Prevalence | Calibrated Accuracy (%) | ROC-AUC (%) | PR-AUC (%) | Top-100 Precision | Top-500 Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$T$ (Diagnostic)** | Synchronous cell occurrence | 50.00% | 58.44% | 62.18% | 60.15% | **87.0%** | **80.0%** |
| **$T+24\text{h}$ (Next-Day)** | Forward cell occurrence | 2.32% | 97.68% | 53.13% | 2.57% | **7.0%** ($3.0\times$ base) | 4.2% |
| **$T+48\text{h}$ (Two-Day)** | Forward cell occurrence | 2.58% | 97.42% | 54.55% | 2.98% | **3.0%** ($1.2\times$ base) | 2.6% |
| **Event Persistence $24\text{h}$** | Connected complex continuity | 6.87% | 93.13% | **68.39%** | **11.90%** | **18.0%** ($2.6\times$ base) | 13.6% |

---

## 5. Controlled Modality Ablation Matrix

| Ablation Stage | Modality Composition | Feature Count | ROC-AUC (%) | PR-AUC (%) | F1-Score (%) | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage A** | Weather 1-Day Only | 6 | 56.25% | 54.37% | 62.51% | 0.2467 | 0.0230 |
| **Stage B** | Weather Multi-Timescale (1d+3d+7d) | 26 | 57.84% (+1.59%) | 55.90% | 61.61% | 0.2451 | 0.0198 |
| **Stage C** | Weather History + Fire History | 28 | 60.53% (+2.69%) | 59.05% | 55.74% | 0.2406 | 0.0226 |
| **Stage D** | Weather History + Real DEM & VPD | 32 | 57.96% | 55.84% | 61.00% | 0.2449 | 0.0205 |
| **Stage E** | **Full Multimodal Integration** | **39** | **62.18% (+5.93%)** | **60.15%** | 59.03% | **0.2375** | **0.0171** |

---

## 6. Multi-Date Historical Replay Benchmark (N=20 Origin Dates)
*Evaluated across 20 dates sampled across peak fire season (Feb–May 2024 and 2025):*
- Total Monitored Cells: 1,832
- Total Verified Fire Hits: 32
- False Alarms: 1,482
- Missed Fire Cells: 7
- **Macro Recall: 66.17%** (captures nearly two out of three next-day fire complexes)
- **Macro Precision: 2.35%** (consistent with the 2.32% natural forward fire prevalence)
- **Macro F1: 4.50%**

---

## 7. Uncertainty Validation & Limitations
- Monte Carlo Dropout predictive standard deviation on the BiGRU deep model produced a mean epistemic uncertainty of $\sigma_{\text{epistemic}} = 0.0207$.
- **Empirical Correlation**: Spearman rank correlation between epistemic uncertainty and absolute classification error was $r = -0.1355$ ($p = 4.76 \times 10^{-129}$).
- **Scientific Interpretation**: Raw MC Dropout standard deviations do not exhibit a positive monotonic relationship with empirical classification error in this dataset, indicating that epistemic variance from dropout alone is insufficient as a standalone error predictor without conformal prediction or distance-to-support bounds.
