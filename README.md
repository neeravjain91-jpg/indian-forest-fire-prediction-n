# Event-Centric Multimodal Spatiotemporal Wildfire Intelligence for India

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20LightGBM-orange.svg)](https://pytorch.org/)
[![Evaluation: Disjoint Holdouts](https://img.shields.io/badge/Evaluation-Spatially%20Disjoint-brightgreen.svg)](docs/EVALUATION_PROTOCOL.md)
[![Testing: Pytest](https://img.shields.io/badge/Tests-Passing%20(52%2F52)-brightgreen.svg)](tests/)

An event-centric, multimodal spatiotemporal wildfire forecasting framework for sovereign India (2018–2025). Integrates **NASA FIRMS VIIRS 375m** satellite telemetry, **Copernicus ERA5-Land** multi-timescale atmospheric reanalysis, authoritative **NOAA ETOPO 2022** digital elevation geomorphology (incorporating NASA SRTM v3), atmospheric fuel moisture deficit dynamics, and connected-component spatiotemporal fire event tracking.

---

## 1. Key Scientific Corrections & Defensible Findings

1. **Authoritative DEM-Derived Topography**:
   - Replaced mathematical approximations with the **NOAA ETOPO 2022 Global Relief Model** (incorporating NASA SRTM v3 land elevation) at 0.10° resolution (~11.1 km) across sovereign India (93,611 grid cells).
   - Topographic slope derived via the canonical 3x3 weighted finite-difference gradient (Horn, 1981); topographic ruggedness derived via Riley et al. (1999) Topographic Ruggedness Index over 8 spatial neighbors.
2. **Strictly Causal Time-Indexed Fire History**:
   - Time-indexed cumulative historical recurrence enforcing strictly $t < T$ causality via binary search. No contemporaneous or future observations enter the historical feature space.
3. **Connected-Component Event Persistence Target**:
   - Fire persistence is redefined at the spatiotemporal complex level: $\text{Target}_{\text{persistence}} = 1$ if and only if an active event continues burning into the subsequent 24-hour cycle within $\le 25\text{ km}$ spatial proximity. Achieves **68.39% ROC-AUC** and **18.0% Top-100 Precision** ($2.6\times$ baseline prevalence).
4. **Controlled 2x2 Factorial Benchmark with Paired Bootstrap 95% Confidence Intervals**:
   - Evaluated under identical splits ($B = 1,000$ paired bootstrap resamples):
     - **Feature Main Effect**: $\Delta \text{ROC-AUC} = +2.89\%$ ($95\% \text{ CI} = [+2.40\%, +3.35\%]$, excludes zero).
     - **Model Family Main Effect**: $\Delta \text{ROC-AUC} = +0.35\%$ ($95\% \text{ CI} = [+0.10\%, +0.56\%]$, excludes zero).
     - **Factorial Interaction**: $\Delta \text{ROC-AUC} = +0.35\%$ ($95\% \text{ CI} = [-0.02\%, +0.73\%]$, crosses zero / not distinguishable).
     - **Calibration Finding**: Parametric Platt scaling regularized probabilities and reduced test-set ECE ($0.0150 \to 0.0143$ in LightGBM 39; $0.0119 \to 0.0088$ in LightGBM 31), whereas non-parametric isotonic regression overfit the validation set and increased test ECE ($0.0171$).
5. **Leave-One-Geographic-Regime-Out (LOGRO) Spatial Cross-Validation**:
   - Systematically stress-tested across six predefined geographic fire regimes (Central, Western Ghats, Northeast, North, East, Northwest) with internal temporal validation (Train $\le 2022$, Val $= 2023$) to prevent spatial autocorrelation leakage.
   - LightGBM Multimodal achieves a Macro Cross-Regional Mean ROC-AUC of **$64.20\% \pm 0.97\%$**, outperforming the 31-feature baseline ($57.46\% \pm 1.66\%$) while cutting cross-regional variance nearly in half.
6. **Historical Replay Benchmark (N=20 Dates)**:
   - Evaluated across 20 origin dates sampled across the held-out test period (Feb–May 2024 and 2025): **66.17% Candidate-Domain Recall** on monitored cells; **3.69% Full Spatial Recall** nationwide (Macro Precision: **2.35%**).

---

## 2. Experimental Benchmark Summary

### Controlled 2x2 Factorial Comparison (Chronological Test 2024–2025)
*Trained on 2018–2022 ($N = 84,661$), Calibrated on 2023 ($N = 14,814$), Tested on 2024–2025 ($N = 31,525$):*

| Experiment ID | Model Architecture | Feature Space | Calibration Protocol | Accuracy (%) | ROC-AUC (%) | PR-AUC (%) | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | HistGradientBoosting (Baseline) | 31 Baseline | Raw (Uncalibrated) | 56.26% | 59.08% | 57.36% | 0.2425 | 0.0092 |
| **Exp A** | HistGradientBoosting (Baseline) | 31 Baseline | Platt Scaling | 56.29% | 59.08% | 57.36% | 0.2426 | 0.0092 |
| **Exp B** | HistGradientBoosting | 39 Multimodal | Platt Scaling | 57.85% | 61.79% | 60.57% | 0.2383 | 0.0173 |
| **Exp C** | LightGBM | 31 Baseline | Platt Scaling | 56.40% | 59.25% | 57.73% | 0.2422 | 0.0088 |
| **Exp D** | **LightGBM (Primary Major)** | **39 Multimodal** | **Platt Scaling** | **58.35%** | **62.31%** | **61.14%** | **0.2371** | **0.0143** |
| *Ref* | Random Forest | 39 Multimodal | Platt Scaling | 57.48% | 61.02% | 59.39% | 0.2403 | 0.0185 |
| *Ref* | Logistic Regression | 39 Multimodal | Platt Scaling | 55.65% | 58.47% | 56.84% | 0.2440 | 0.0115 |
| *Ref* | Multi-Scale BiGRU Deep Net | Modality Tensors | Raw | 55.91% | 58.92% | 57.61% | 0.2434 | 0.0190 |

---

## 3. Installation & Reproducibility

### 1. Setup Environment
```bash
git clone https://github.com/neeravjain91-jpg/major-forest-fire.git
cd major-forest-fire
pip install -r requirements.txt
```

### 2. Reproduce the Corrected Research Pipeline
```bash
# 1. Build multimodal dataset with real DEM, causal history, and LOGRO splits
python -m src.data.dataset_builder

# 2. Run controlled 2x2 factorial baseline suite with bootstrap 95% CIs and calibration comparison
python -m src.models.baselines

# 3. Train multi-scale temporal BiGRU deep model
python -m src.models.multimodal_deep --epochs 20

# 4. Run Leave-One-Geographic-Regime-Out (LOGRO) cross-validation across all 6 regimes
python -m src.evaluation.geographic_eval

# 5. Evaluate multi-horizon forward forecasting (T, T+24h, T+48h, Persistence)
python -m src.models.multi_horizon_eval

# 6. Execute controlled modality ablation study
python -m src.models.ablation_study

# 7. Run 20-date historical replay verification benchmark
python -m src.replay.historical_replay

# 8. Generate publication-ready figures
python -m src.evaluation.generate_figures

# 9. Run automated test suite
pytest -v
```

### 3. Launch the Intelligence Platform
```bash
python application.py
```
Access at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 4. Scientific Documentation Index

- [docs/CURRENT_STATE_AUDIT.md](docs/CURRENT_STATE_AUDIT.md): Mini-to-major architectural transition audit.
- [docs/MAJOR_PROJECT_ARCHITECTURE.md](docs/MAJOR_PROJECT_ARCHITECTURE.md): Research system architecture and data flows.
- [docs/LITERATURE_GAP.md](docs/LITERATURE_GAP.md): Survey of prior literature through 2026 and gap analysis.
- [reports/literature_matrix.csv](reports/literature_matrix.csv): Literature benchmarking matrix.
- [docs/RESEARCH_QUESTIONS.md](docs/RESEARCH_QUESTIONS.md): Formal research hypotheses and statistical verification criteria.
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md): Authoritative documentation for VIIRS, ERA5, and NOAA DEM.
- [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md): Complete feature dictionary and target definitions.
- [docs/LEAKAGE_POLICY.md](docs/LEAKAGE_POLICY.md): Temporal causality and spatial leakage control rules.
- [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md): Mathematical formulations of metrics, LOGRO, and bootstrap CIs.
- [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md): Detailed registry of all experiment runs.
- [docs/RESULTS.md](docs/RESULTS.md): Empirical result tables.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): Production deployment and security guide.

---

## 5. Scientific Integrity & Limitations

1. **Defensible Forecast Horizons**: Because VIIRS active fire observations over India are acquired via polar sun-synchronous satellites (~2 overpasses per 24 hours), continuous hourly predictions ($T+1\text{h}$) are scientifically unsupportable without geostationary sensors. Defensible horizons are formulated at **$T+24\text{h}$** and **$T+48\text{h}$**.
2. **Negative Label Caveat**: Absence of a satellite detection on day $T+1\text{d}$ denotes non-detection during overpasses, not omniscience of sub-canopy ground conditions.
3. **Surveillance vs. Population Incidence**: The underlying dataset is a 1:1 case-control retrospective sample. Model probabilities represent sample odds; true daily spatial occurrence is $< 0.05\%$ across India.
4. **Research Interface**: The web platform is a research demonstration and spatial verification benchmark, **not an operational emergency warning system**.
