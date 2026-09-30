# Experimental Results & Rigorous Hypothesis Verification

All findings presented herein are derived from empirical observations from the 131,000-sample India wildfire dataset (2018–2025) using the **NOAA ETOPO 2022 Global Relief Model** (incorporating NASA SRTM v3 land elevation), strictly causal time-indexed fire history ($t < T$), and connected-component spatiotemporal event persistence targets.

---

## 1. Controlled 2x2 Factorial Baseline Comparison (Chronological Test 2024–2025)

*Dataset Splits*:
- Training Set: 2018–2022 ($N = 84,661$)
- Validation Set: 2023 ($N = 14,814$, used strictly for calibrator fitting and model selection)
- Held-Out Test Set: 2024–2025 ($N = 31,525$, untouched during training and calibration)

### Empirical Evaluation Across Calibration Protocols (Raw vs. Platt vs. Isotonic)

| Experiment ID | Model Architecture | Features | Calibration Protocol | Accuracy (%) | F1 (%) | ROC-AUC (%) | PR-AUC (%) | Brier Score | ECE | MCE |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | HistGradientBoosting | 31 | Raw (Uncalibrated) | 56.26% | 57.55% | 59.08% | 57.36% | 0.2425 | 0.0092 | 0.0661 |
| **Exp A** | HistGradientBoosting | 31 | Platt Scaling | 56.29% | 59.95% | 59.08% | 57.36% | 0.2426 | 0.0092 | 0.0872 |
| **Exp A** | HistGradientBoosting | 31 | Isotonic Regression | 55.85% | 63.47% | 59.03% | 56.74% | 0.2427 | 0.0137 | 0.1227 |
| **Exp B** | HistGradientBoosting | 39 | Raw (Uncalibrated) | 57.72% | 56.81% | 61.79% | 60.57% | 0.2381 | 0.0132 | 0.0422 |
| **Exp B** | HistGradientBoosting | 39 | Platt Scaling | 57.85% | 58.67% | 61.79% | 60.57% | 0.2383 | 0.0173 | 0.0414 |
| **Exp B** | HistGradientBoosting | 39 | Isotonic Regression | 57.85% | 59.52% | 61.73% | 59.68% | 0.2384 | 0.0177 | 0.1167 |
| **Exp C** | LightGBM | 31 | Raw (Uncalibrated) | 56.39% | 57.18% | 59.25% | 57.73% | 0.2422 | 0.0119 | 0.0350 |
| **Exp C** | LightGBM | 31 | Platt Scaling | 56.40% | 59.78% | 59.25% | 57.73% | 0.2422 | **0.0088** | 0.0375 |
| **Exp C** | LightGBM | 31 | Isotonic Regression | 56.33% | 62.47% | 59.23% | 57.00% | 0.2424 | 0.0153 | 0.0622 |
| **Exp D** | **LightGBM (Primary Major)** | **39** | Raw (Uncalibrated) | 58.10% | 56.22% | **62.31%** | **61.14%** | **0.2371** | 0.0150 | 0.0682 |
| **Exp D** | **LightGBM (Primary Major)** | **39** | **Platt Scaling** | **58.35%** | 60.06% | **62.31%** | **61.14%** | **0.2371** | **0.0143** | 0.0922 |
| **Exp D** | **LightGBM (Primary Major)** | **39** | Isotonic Regression | 58.44% | 59.03% | 62.18% | 60.15% | 0.2375 | 0.0171 | 0.0749 |
| *Ref* | Random Forest | 39 | Platt Scaling | 57.48% | 56.45% | 61.02% | 59.39% | 0.2403 | 0.0185 | 0.0775 |
| *Ref* | Logistic Regression | 39 | Platt Scaling | 55.65% | 57.90% | 58.47% | 56.84% | 0.2440 | 0.0115 | 0.1318 |
| *Ref* | Multi-Scale BiGRU Deep Net | Tensors | Raw | 55.91% | 57.14% | 58.92% | 57.61% | 0.2434 | 0.0190 | 0.0820 |

> **Critical Calibration Finding**: Post-hoc isotonic regression overfit the 2023 validation distribution, resulting in an **increase** in test-set ECE across all models ($0.0092 \to 0.0137$ in HGB 31; $0.0150 \to 0.0171$ in LightGBM 39). In contrast, parametric Platt scaling (logistic calibration on log-odds) regularized probability estimates and reduced test-set ECE ($0.0150 \to 0.0143$ in LightGBM 39; $0.0119 \to 0.0088$ in LightGBM 31). Brier scores consistently improved under multimodal feature expansion, whereas ECE improvement was method-dependent.

---

## 2. True 2x2 Factorial Interaction & Paired Bootstrap Analysis

*Design Matrix*:
- Cell A: HistGradientBoosting + 31 Baseline Features
- Cell B: HistGradientBoosting + 39 Multimodal Features
- Cell C: LightGBM + 31 Baseline Features
- Cell D: LightGBM + 39 Multimodal Features

*Evaluated via paired bootstrap resampling ($B = 1,000$) of the identical test observations ($N = 31,525$):*

| Effect Name | Metric | Observed Effect ($\Delta$) | 95% Bootstrap Confidence Interval | Excludes Zero? | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Feature Main Effect** | ROC-AUC | **+0.0289 (+2.89%)** | **[+0.0240, +0.0335]** | **True** | Multimodal features provide substantial gain |
| **Model Family Main Effect** | ROC-AUC | +0.0035 (+0.35%) | [+0.0010, +0.0056] | **True** | Small algorithm advantage for LightGBM |
| **Factorial Interaction** | ROC-AUC | +0.0035 (+0.35%) | [-0.0002, +0.0073] | **False** | Interaction effect crosses zero for ROC-AUC |
| **Feature Main Effect** | PR-AUC | **+0.0331 (+3.31%)** | **[+0.0272, +0.0392]** | **True** | Multimodal features provide substantial gain |
| **Model Family Main Effect** | PR-AUC | +0.0047 (+0.47%) | [+0.0020, +0.0072] | **True** | Modest algorithm advantage for LightGBM |
| **Factorial Interaction** | PR-AUC | +0.0019 (+0.19%) | [-0.0023, +0.0059] | **False** | Interaction effect crosses zero for PR-AUC |
| **Feature Main Effect** | Brier Score | **-0.0047** | **[-0.0056, -0.0038]** | **True** | Significant probability error reduction |
| **Model Family Main Effect** | Brier Score | -0.0007 | [-0.0011, -0.0004] | **True** | Minor error reduction from LightGBM |
| **Factorial Interaction** | Brier Score | **-0.0008** | **[-0.0014, -0.0002]** | **True** | LightGBM extracts greater Brier reduction from 39 feats |
| **Factorial Interaction** | F1-Score | **+0.0157 (+1.57%)** | **[+0.0095, +0.0212]** | **True** | Statistically distinguishable interaction in F1 |

> **Neutral Scientific Synthesis**: The multimodal feature expansion accounts for the dominant observed performance increment ($\Delta \text{ROC-AUC} = +2.89\%$, $\Delta \text{PR-AUC} = +3.31\%$), while model-family differences are substantially smaller ($\Delta \text{ROC-AUC} = +0.35\%$) and feature-dependent.

---

## 3. Leave-One-Geographic-Regime-Out (LOGRO) Spatial Cross-Validation

*Protocol*: The model is trained on 5 geographic regimes (with internal temporal validation: Train $\le 2022$, Val $= 2023$ to eliminate spatial autocorrelation leakage) and tested exclusively on the 6th held-out geographic regime across all years:

| Held-Out Geographic Regime | Regional Setting | HGB (31 Baseline) ROC-AUC (%) | LightGBM (39 Multimodal) ROC-AUC (%) | $\Delta$ ROC-AUC (%) | HGB Brier Score | LGBM Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CENTRAL** | Interior Deccan dry deciduous plateau | 57.67% | **64.55%** | **+6.88%** | 0.2448 | **0.2336** |
| **WESTERN_GHATS** | Moist western coastal escarpment | 55.71% | **63.70%** | **+7.99%** | 0.2481 | **0.2354** |
| **NORTHEAST** | Purvanchal & Brahmaputra basin | 60.28% | **65.65%** | **+5.37%** | 0.2409 | **0.2336** |
| **NORTH** | Himalayan montane & foothills | 57.35% | **64.10%** | **+6.75%** | 0.2459 | **0.2354** |
| **EAST** | Eastern Ghats / Chota Nagpur | 57.88% | **62.74%** | **+4.86%** | 0.2454 | **0.2377** |
| **NORTHWEST** | Semi-arid Thar & Aravalli scrub | 55.89% | **64.49%** | **+8.60%** | 0.2472 | **0.2338** |
| **Macro Cross-Regional Mean** | **National Aggregate** | **57.46% ± 1.66%** | **64.20% ± 0.97%** | **+6.74%** | **0.2454** | **0.2349** |

*Methodological Clarification*: These regions represent predefined latitudinal-longitudinal macro-climatic partitions, rather than official WWF Terrestrial Ecoregions or WII biogeographic boundary polygons. LightGBM 39 Multimodal reduces cross-regional variance from $\pm 1.66\%$ to $\pm 0.97\%$ while outperforming the baseline across all six held-out regimes.

---

## 4. Multi-Horizon Forecasting & Rare-Event Precision@k

| Horizon | Target Definition | Positive Prevalence | Calibrated Accuracy (%) | ROC-AUC (%) | PR-AUC (%) | Top-100 Precision | Top-500 Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T$ (Diagnostic)** | Synchronous active-fire occurrence | 50.00% | 58.35% | 62.31% | 61.14% | **87.0%** | **80.0%** |
| **$T+24\text{h}$ (Next-Day)** | Forward cell thermal anomaly | 2.32% | 97.68% | 53.13% | 2.57% | **7.0%** ($3.0\times$ base) | 4.2% |
| **$T+48\text{h}$ (Two-Day)** | Forward cell thermal anomaly | 2.58% | 97.42% | 54.55% | 2.98% | **3.0%** ($1.2\times$ base) | 2.6% |
| **Event Persistence $24\text{h}$** | Connected complex continuity | 6.87% | 93.13% | **68.39%** | **11.90%** | **18.0%** ($2.6\times$ base) | **13.6%** |

### Honest Scientific Assessment of Forward Horizons:
1. **Forward Grid-Cell Occurrence ($T+24\text{h}$, $T+48\text{h}$)**: Predicting new grid-cell thermal anomalies 24–48 hours in advance yields an ROC-AUC of only $53.13\%$ and PR-AUC of $2.57\%$. While top-ranked predictions ($k=100$) achieve $7.0\%$ precision ($3\times$ the natural $2.32\%$ base rate), overall spatial discrimination from synoptic meteorology and topography alone is near-random ($53\%$) without real-time human activity, land-use management, or lightning telemetry.
2. **Event Persistence ($24\text{h}$)**: Conversely, predicting the forward persistence of an **already-active connected fire complex** achieves **$68.39\%$ ROC-AUC** and **$18.0\%$ Top-100 Precision** ($2.6\times$ baseline). Modeling wildfire at the event-complex scale is a vastly better-posed scientific formulation than predicting arbitrary cell ignitions.

---

## 5. Controlled Modality Ablation Matrix

| Ablation Stage | Modality Composition | Feature Count | ROC-AUC (%) | PR-AUC (%) | F1-Score (%) | Brier Score | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage A** | Weather 1-Day Only | 6 | 56.25% | 54.37% | 62.51% | 0.2467 | 0.0230 |
| **Stage B** | Weather Multi-Timescale (1d+3d+7d) | 26 | 57.84% (+1.59%) | 55.90% | 61.61% | 0.2451 | 0.0198 |
| **Stage C** | Weather History + Causal Fire History | 28 | 60.53% (+2.69%) | 59.05% | 55.74% | 0.2406 | 0.0226 |
| **Stage D** | Weather History + Real DEM & Fuel Proxy | 32 | 57.96% | 55.84% | 61.00% | 0.2449 | 0.0205 |
| **Stage E** | **Full Multimodal Integration** | **39** | **62.18% (+5.93%)** | **60.15%** | 59.03% | **0.2375** | **0.0171** |

---

## 6. Historical Replay Benchmark: Candidate Domain vs. Full Spatial Domain

*Evaluated across 20 origin dates sampled across the peak fire seasons of 2024 and 2025:*

- Total Candidate Cells Monitored: 1,832
- Total Verified Fire Hits: 32
- Total Candidate False Alarms: 1,482
- Total Confirmed Nationwide Fires at $T+1\text{d}$: 879
- **Candidate-Domain Recall (conditioned on monitored cells)**: **$66.17\%$**
- **Full Spatial Domain Recall (nationwide denominator)**: **$3.69\%$**
- **Macro Precision**: **$2.35\%$** (empirically aligning with the $2.32\%$ test-set forward fire prevalence)
- **Macro False Positive Rate**: $33.51\%$

> **Domain Construction Disclosure**: Reporting $66.17\%$ recall is valid only when conditioned on the sampled candidate cells active at time $T$. When evaluated against all active fire pixels detected across sovereign India on $T+1\text{d}$, the unconditioned spatial recall is $3.69\%$. This illustrates the profound difference between surveillance tracking and unconstrained nationwide detection.

---

## 7. Data Provenance & Observational Caveats

1. **DEM Provenance**: All topographic derivatives are derived from the **NOAA ETOPO 2022 Global Relief Model (v1, 15 arc-second grid)**, which embeds NASA SRTM v3.0 land elevation. Topographic slope is calculated via the canonical 3x3 weighted finite-difference gradient (Horn, 1981), and topographic ruggedness is derived via Riley et al. (1999) Topographic Ruggedness Index over 8 spatial neighbors.
2. **Case-Control Sampling Structure**: The 131,000-sample dataset is a 1:1 retrospective case-control sample ($P(Y=1) = 0.50$ at time $T$). Model probabilities reflect sample odds, not population incidence (which is $< 0.05\%$ per grid cell daily across India).
3. **Negative Label Meaning**: In satellite remote sensing, an active fire detection ($Y=1$) is a confirmed VIIRS thermal anomaly. A negative label ($Y=0$) denotes the absence of confirmed thermal anomaly detection during satellite overpasses (which may occur due to cloud obscuration, sub-pixel fire size, or overpass timing), not guaranteed physical absence of ground combustion.
4. **Epistemic Uncertainty Limitation**: Spearman rank correlation between MC Dropout standard deviation and empirical absolute classification error was $r_s = -0.1355$ ($p < 0.001$). Dropout variance alone does not provide a monotonic error detector without spatial distance-to-support bounds.
