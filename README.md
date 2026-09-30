# Event-Centric Multimodal Spatiotemporal Wildfire Intelligence for India

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20LightGBM-orange.svg)](https://pytorch.org/)
[![Evaluation: Disjoint Holdouts](https://img.shields.io/badge/Evaluation-Spatially%20Disjoint-brightgreen.svg)](docs/EVALUATION_PROTOCOL.md)
[![Testing: Pytest](https://img.shields.io/badge/Tests-Passing%20(7%2F7)-brightgreen.svg)](tests/test_wildfire_research.py)

An event-centric, multimodal spatiotemporal wildfire forecasting framework for sovereign India (2018–2025). Integrates **NASA FIRMS VIIRS 375m** satellite telemetry, **Copernicus ERA5-Land** multi-timescale atmospheric reanalysis, authoritative **NOAA ETOPO / SRTM** digital elevation geomorphology, atmospheric fuel dryness dynamics, and connected-component spatiotemporal fire event tracking.

---

## 1. Key Scientific Corrections & Defensible Findings

1. **Real DEM-Derived Topography**:
   - Replaced mathematical approximations with authoritative **NOAA ETOPO 2022 / NASA SRTM DEM** elevation data across the entire Indian subcontinent ($6^\circ\text{N} - 37^\circ\text{N}$, $68^\circ\text{E} - 98^\circ\text{E}$).
   - Topographic slope derived via Horn's spatial finite-difference gradient; topographic ruggedness derived via Riley Topographic Ruggedness Index (TRI).
2. **Causal Time-Indexed Fire History**:
   - Time-indexed cumulative historical recurrence enforcing strictly $t < T$ causality via binary search. No contemporaneous or future observations enter the historical feature space.
3. **Connected-Component Event Persistence Target**:
   - Fire persistence is redefined at the spatiotemporal complex level: $\text{Target}_{\text{persistence}} = 1$ only if the same connected spatiotemporal event continues burning into the subsequent 24-hour cycle. Achieves **68.39% ROC-AUC** and **18.0% Top-100 Precision** ($2.6\times$ baseline prevalence).
4. **Controlled 2x2 Factorial Benchmark with Bootstrap 95% Confidence Intervals**:
   - Evaluated under identical splits and post-hoc isotonic calibration ($B = 1,000$ bootstrap resamples):
     - **Multimodal Effect in HGB**: $\Delta \text{ROC-AUC} = +2.71\%$ ($95\% \text{ CI} = [+2.16\%, +3.22\%]$, excludes zero).
     - **Multimodal Effect in LightGBM**: $\Delta \text{ROC-AUC} = +2.95\%$ ($95\% \text{ CI} = [+2.42\%, +3.44\%]$, excludes zero).
     - **Algorithm Effect on Baseline Features**: $\Delta \text{ROC-AUC} = +0.21\%$ ($95\% \text{ CI} = [-0.14\%, +0.53\%]$, crosses zero / not distinguishable).
5. **Leave-One-Ecoregion-Out (LOEO) Geographic Cross-Validation**:
   - Systematically stress-tested across all 6 Indian biomes (Central, Western Ghats, Northeast, North, East, Northwest).
   - LightGBM Multimodal achieves a Macro Cross-Regional Mean ROC-AUC of **$64.58\% \pm 0.95\%$**, outperforming the 31-feature baseline ($57.75\% \pm 1.96\%$) with lower cross-regional variance.
6. **Multi-Date Historical Replay Benchmark (N=20 Dates)**:
   - Evaluated across 20 origin dates sampled across the held-out test period (Feb–May 2024 and 2025): **66.17% Macro Recall** on next-day fire complexes.

---

## 2. Experimental Benchmark Summary

### Controlled 2x2 Factorial Comparison (Chronological Test 2024–2025)
*Trained on 2018–2022 ($N = 84,661$), Calibrated on 2023 ($N = 14,814$), Tested on 2024–2025 ($N = 31,525$):*

| Experiment ID | Model Architecture | Feature Space | Calibrated Acc (%) | Calibrated ROC-AUC (%) | Calibrated PR-AUC (%) | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp A** | HistGradientBoosting (Mini Baseline) | 31 Baseline | 55.85% | 59.03% | 56.74% | 0.2427 | 0.0137 |
| **Exp B** | HistGradientBoosting | 39 Multimodal | 57.85% | 61.73% | 59.68% | 0.2384 | 0.0177 |
| **Exp C** | LightGBM | 31 Baseline | 56.33% | 59.23% | 57.00% | 0.2424 | 0.0153 |
| **Exp D** | **LightGBM (Primary Major)** | **39 Multimodal** | **58.44%** | **62.18%** | **60.15%** | **0.2375** | 0.0171 |
| *Ref* | Random Forest | 39 Multimodal | 57.51% | 60.94% | 58.51% | 0.2402 | 0.0172 |
| *Ref* | Logistic Regression | 39 Multimodal | 55.97% | 58.40% | 56.18% | 0.2446 | 0.0225 |
| *Ref* | Spatiotemporal BiGRU Deep Net | Modality Tensors | 55.91% | 58.92% | 57.61% | 0.2434 | 0.0190 |

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
# 1. Build multimodal dataset with real DEM, causal history, and LOEO splits
python -m src.data.dataset_builder

# 2. Run controlled 2x2 factorial baseline suite with bootstrap 95% CIs
python -m src.models.baselines

# 3. Train temporal BiGRU spatiotemporal deep model
python -m src.models.multimodal_deep --epochs 20

# 4. Run Leave-One-Ecoregion-Out (LOEO) cross-validation across all 6 biomes
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
pytest -v tests/test_wildfire_research.py
```

### 3. Launch the Intelligence Platform
```bash
python application.py
```
Access at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 4. Scientific Documentation Index

- [`docs/CURRENT_STATE_AUDIT.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/CURRENT_STATE_AUDIT.md): Mini-to-major architectural transition audit.
- [`docs/MAJOR_PROJECT_ARCHITECTURE.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/MAJOR_PROJECT_ARCHITECTURE.md): Research system architecture and data flows.
- [`docs/LITERATURE_GAP.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/LITERATURE_GAP.md): Survey of prior literature through 2026 and gap analysis.
- [`reports/literature_matrix.csv`](file:///c:/Users/ASUS/Downloads/forest%20fire/reports/literature_matrix.csv): Literature benchmarking matrix.
- [`docs/RESEARCH_QUESTIONS.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/RESEARCH_QUESTIONS.md): Formal research hypotheses and statistical verification criteria.
- [`docs/DATA_SOURCES.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/DATA_SOURCES.md): Authoritative documentation for VIIRS, ERA5, and NOAA DEM.
- [`docs/DATA_SCHEMA.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/DATA_SCHEMA.md): Complete feature dictionary and target definitions.
- [`docs/LEAKAGE_POLICY.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/LEAKAGE_POLICY.md): Temporal causality and spatial leakage control rules.
- [`docs/EVALUATION_PROTOCOL.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/EVALUATION_PROTOCOL.md): Mathematical formulations of metrics, LOEO, and bootstrap CIs.
- [`docs/EXPERIMENT_MATRIX.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/EXPERIMENT_MATRIX.md): Detailed registry of all experiment runs.
- [`docs/RESULTS.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/RESULTS.md): Empirical result tables.
- [`docs/DEPLOYMENT.md`](file:///c:/Users/ASUS/Downloads/forest%20fire/docs/DEPLOYMENT.md): Production deployment and security guide.

---

## 5. Scientific Integrity & Limitations

1. **Defensible Forecast Horizons**: Because VIIRS active fire observations over India are acquired via polar sun-synchronous satellites (~2 overpasses per 24 hours), continuous hourly predictions ($T+1\text{h}$) are scientifically unsupportable without geostationary sensors. Defensible horizons are formulated at **$T+24\text{h}$** and **$T+48\text{h}$**.
2. **Negative Label Caveat**: Absence of a satellite detection on day $T+1\text{d}$ denotes non-detection during overpasses, not omniscience of sub-canopy ground conditions.
3. **Research Interface**: The web platform is a research demonstration and spatial verification benchmark, **not an operational emergency warning system**.
