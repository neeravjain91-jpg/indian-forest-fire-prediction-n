# Event-Centric Multimodal Spatiotemporal Wildfire Intelligence for India

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch | LightGBM](https://img.shields.io/badge/Framework-PyTorch%20%7C%20LightGBM-orange.svg)](https://pytorch.org/)
[![Evaluation: Disjoint Holdouts](https://img.shields.io/badge/Evaluation-Spatially%20Disjoint-brightgreen.svg)](docs/EVALUATION_PROTOCOL.md)
[![Testing: Pytest](https://img.shields.io/badge/Tests-Passing%20(52%2F52)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An event-centric, multimodal spatiotemporal wildfire forecasting framework for sovereign India (2018–2025). Integrates **NASA FIRMS VIIRS 375m** satellite telemetry, **Copernicus ERA5-Land** multi-timescale atmospheric reanalysis, authoritative **NOAA ETOPO 2022** digital elevation geomorphology (incorporating NASA SRTM v3), atmospheric fuel moisture deficit dynamics, and connected-component spatiotemporal fire event tracking.

---

## 1. Problem
Wildfire prediction in India faces severe observational and environmental constraints:
- **Observation Sparsity**: Polar sun-synchronous satellites (VIIRS 375m) provide approximately two observations per 24 hours, making continuous hourly spread forecasting unsupportable without geostationary instruments.
- **Topographic & Ecological Heterogeneity**: India spans diverse biophysical regimes (dry deciduous plateaus, tropical evergreen Ghats, Himalayan montane slopes, semi-arid scrub), making static location-dependent models prone to overfitting.
- **Occurrence vs. Event Persistence**: Predicting where arbitrary single-cell ignitions will happen 24–48 hours ahead from synoptic meteorology is poorly posed, whereas tracking whether already-ignited fire complexes continue burning is physically governed and operationally actionable.

## 2. Research Questions & Hypotheses
This project investigates whether incorporating multi-timescale meteorology, causal fire history, and authoritative terrain geomorphology yields measurable improvements in forward forecasting, probability calibration, and geographic transfer:
- **H1 (Temporal Meteorology)**: Antecedent multi-timescale weather (1d, 3d, 7d) improves forward prediction over static single-day snapshots.
- **H2 (Event Representation)**: Connected-component event persistence provides higher signal ($68.39\%$ ROC-AUC) than arbitrary single-cell forward ignition forecasting ($53.13\%$).
- **H3 (Multimodal Transfer)**: Multimodal features improve spatial generalization across six held-out geographic regimes over location/weather-only models.
- **H4 (Calibration & Uncertainty)**: Post-hoc probability calibration distinguishes Brier score error reduction from non-parametric calibration degradation under distribution shifts.

Detailed formalizations are available in [docs/RESEARCH_QUESTIONS.md](docs/RESEARCH_QUESTIONS.md).

## 3. Scientific Architecture
The system employs an event-centric spatiotemporal architecture:
1. **Multimodal Feature Backbone**: Encompasses coordinates, multi-timescale meteorology, vapor pressure deficit (VPD), topsoil drought index, canonical terrain derivatives, and causal historical recurrence ($t < T$).
2. **Multi-Scale Temporal Feature Encoder**: BiGRU temporal recurrent model processing sequenced 1d, 3d, and 7d atmospheric drying vectors alongside tabular topographic embeddings.
3. **Controlled 2x2 Factorial Matrix**: Systematically isolates model family effects (HistGradientBoosting vs. LightGBM) from feature composition effects (31-feature baseline vs. 39-feature multimodal).

Architecture specifications and dataflow diagrams are documented in [docs/MAJOR_PROJECT_ARCHITECTURE.md](docs/MAJOR_PROJECT_ARCHITECTURE.md).

## 4. Authoritative Data Sources
All datasets adhere to strict provenance and open scientific licensing:
- **Active Fire Telemetry**: VIIRS 375m NRT (`VNP14IMGTDL_NRT`, `VJ114IMGTDL_NRT`, `VJ214IMGTDL_NRT`) via NASA FIRMS.
- **Atmospheric Reanalysis**: Copernicus ERA5-Land hourly reanalysis at 0.10° resolution, aggregated into antecedent 24h, 72h, and 168h windows.
- **Terrain Geomorphology**: **NOAA ETOPO 2022 Global Relief Model** (15 arc-second native, embedding NASA SRTM v3.0 land elevation), bilinearly resampled to 0.10° across 93,611 grid cells of sovereign India.
- **Administrative Boundaries**: Survey of India official national boundary vector polygons.

Complete provider citations and preprocessing pipelines are detailed in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## 5. Methodology
- **Topographic Derivatives**:
  - Slope computed via canonical 3x3 weighted finite-difference gradient (Horn, 1981).
  - Topographic Ruggedness Index (TRI) computed via Riley et al. (1999) 8-neighbor root-sum-square elevation variance.
- **Causal Fire History**: Binary search on strictly historical records ($t < T$) prevents contemporaneous or future fire detection leakage.
- **Event Clustering & Persistence**: Connected-component spatiotemporal clustering ($\text{DBSCAN-ST}$). Event persistence requires active cluster continuation on calendar date $T+1\text{d}$ within $\le 25\text{ km}$ spatial proximity.
- **Fuel Dryness Proxies**: Daily vapor pressure deficit (VPD) calculated via the Tetens formula; topsoil drought index calculated relative to a nominal $0.35\text{ m}^3/\text{m}^3$ reference threshold.

Full feature schemas and mathematical definitions are cataloged in [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

## 6. Experimental Protocols
- **Strict Chronological Splits**: Partitioned to eliminate lookahead bias:
  - Training: 2018–2022 ($N = 84,661$)
  - Validation: 2023 ($N = 14,814$, used strictly for model selection and calibrator fitting)
  - Held-Out Test: 2024–2025 ($N = 31,525$, evaluated once)
- **Leave-One-Geographic-Regime-Out (LOGRO)**: Evaluates out-of-region generalization across six predefined geographic fire regimes (Central, Western Ghats, Northeast, North, East, Northwest) with internal temporal validation (Train $\le 2022$, Val $= 2023$) to prevent spatial autocorrelation leakage.
- **Statistical Significance**: $B = 1,000$ paired bootstrap resamples on identical test observations computing empirical 95% confidence intervals for all metric deltas.
- **Calibration Comparison**: Raw vs. Platt (logistic) vs. Isotonic (non-parametric) evaluated on untouched test data.

Evaluation metrics and leakage controls are defined in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) and [docs/LEAKAGE_POLICY.md](docs/LEAKAGE_POLICY.md).

## 7. Main Results

### Controlled 2x2 Factorial Benchmark (Test 2024–2025)
*Evaluated on $N = 31,525$ identical test observations:*

| Experiment ID | Model Family | Features | Calibration | Accuracy (%) | F1 (%) | ROC-AUC (%) | PR-AUC (%) | Brier Score | ECE |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | HistGradientBoosting (Baseline) | 31 | Platt | 56.29% | 59.95% | 59.08% | 57.36% | 0.2426 | 0.0092 |
| **Exp B** | HistGradientBoosting | 39 | Platt | 57.85% | 58.67% | 61.79% | 60.57% | 0.2383 | 0.0173 |
| **Exp C** | LightGBM | 31 | Platt | 56.40% | 59.78% | 59.25% | 57.73% | 0.2422 | **0.0088** |
| **Exp D** | **LightGBM (Primary Major)** | **39** | **Platt** | **58.35%** | **60.06%** | **62.31%** | **61.14%** | **0.2371** | **0.0143** |
| *Ref* | Random Forest | 39 | Platt | 57.48% | 56.45% | 61.02% | 59.39% | 0.2403 | 0.0185 |
| *Ref* | Logistic Regression | 39 | Platt | 55.65% | 57.90% | 58.47% | 56.84% | 0.2440 | 0.0115 |
| *Ref* | Multi-Scale BiGRU Deep Net | Tensors | Raw | 55.91% | 57.14% | 58.92% | 57.61% | 0.2434 | 0.0190 |

### Factorial Interaction Analysis (1,000 Paired Bootstrap Resamples)
- **Feature Main Effect**: $\Delta \text{ROC-AUC} = +2.89\%$ ($95\% \text{ CI} = [+2.40\%, +3.35\%]$, excludes zero).
- **Model Family Main Effect**: $\Delta \text{ROC-AUC} = +0.35\%$ ($95\% \text{ CI} = [+0.10\%, +0.56\%]$, excludes zero).
- **Factorial Interaction**: $\Delta \text{ROC-AUC} = +0.35\%$ ($95\% \text{ CI} = [-0.02\%, +0.73\%]$, crosses zero).
- **Calibration Finding**: Parametric Platt scaling regularized probability estimates and reduced test-set ECE ($0.0150 \to 0.0143$ in LightGBM 39; $0.0119 \to 0.0088$ in LightGBM 31), whereas non-parametric isotonic regression overfit the validation set, increasing test ECE ($0.0171$).

### LOGRO Spatial Cross-Validation
- **LightGBM 39 Multimodal**: Macro Cross-Regional Mean ROC-AUC of **$64.20\% \pm 0.97\%$** across all six held-out regimes.
- **HGB 31 Baseline**: Macro Mean ROC-AUC of **$57.46\% \pm 1.66\%$**. Multimodal features cut spatial variance nearly in half while improving transfer across every region.

### Multi-Horizon Discrimination & Historical Replay
- Arbitrary cell forward occurrence ($T+24\text{h}$) from synoptic meteorology and terrain alone yields an ROC-AUC of $53.13\%$ and PR-AUC of $2.57\%$.
- Connected-component **event persistence ($24\text{h}$)** achieves **$68.39\%$ ROC-AUC**, **$11.90\%$ PR-AUC**, and **$18.0\%$ Top-100 Precision** ($2.6\times$ baseline prevalence).
- Historical Replay across 20 dates yields **$66.17\%$ Candidate-Domain Recall** on active candidate cells and **$3.69\%$ Full Spatial Recall** nationwide (Macro Precision: $2.35\%$).

Full metric tables and ablations are reported in [docs/RESULTS.md](docs/RESULTS.md) and [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md).

## 8. Limitations & Methodological Disclosures
1. **Retrospective Case-Control Design**: The dataset is a 1:1 case-control sample ($P(Y=1) = 0.50$ at reference time $T$). Model probabilities reflect sample odds; unadjusted nationwide daily incidence is $< 0.05\%$ per cell.
2. **Satellite Observational Bounds**: $Y=0$ denotes the absence of a confirmed VIIRS thermal detection during satellite overpasses, constrained by cloud cover, canopy obstruction, and sensor resolution ($375\text{ m}$). It does not guarantee the complete physical absence of sub-canopy smoldering.
3. **Forward Ignition Limits**: Predicting arbitrary cell ignitions 24–48 hours ahead without real-time lightning telemetry, human land-use activity, or power-line monitoring remains inherently ill-posed ($53\%$ ROC-AUC).
4. **Research Demonstration Platform**: The web interface is strictly an academic research demonstration and spatial verification station, **not an operational early warning, civil defence, or disaster response system**.

## 9. Reproduction Commands

### Environment Setup
```bash
git clone https://github.com/neeravjain91-jpg/major-forest-fire.git
cd major-forest-fire
pip install -r requirements.txt
```

### Reproduce Research Experiments
```bash
# 1. Build multimodal dataset with DEM, causal history, and LOGRO splits
python -m src.data.dataset_builder

# 2. Run controlled 2x2 factorial baseline suite with bootstrap CIs and calibration
python -m src.models.baselines

# 3. Train multi-scale temporal BiGRU deep model
python -m src.models.multimodal_deep --epochs 20

# 4. Run Leave-One-Geographic-Regime-Out (LOGRO) cross-validation
python -m src.evaluation.geographic_eval

# 5. Evaluate multi-horizon targets (T, T+24h, T+48h, Event Persistence)
python -m src.models.multi_horizon_eval

# 6. Execute modality ablation study
python -m src.models.ablation_study

# 7. Run 20-date prospective historical replay verification benchmark
python -m src.replay.historical_replay

# 8. Generate publication-ready figures
python -m src.evaluation.generate_figures

# 9. Execute automated test suite (52 tests)
pytest -v
```

### Launch Demonstration Application
```bash
python application.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Operational deployment details are provided in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## 10. Project Structure
```
major-forest-fire/
├── README.md                          # Concise project entry point & research summary
├── LICENSE                            # MIT Open-Source License
├── requirements.txt                   # Frozen Python environment dependencies
├── pyproject.toml                     # Project metadata & pytest configuration
├── application.py                     # Research demonstration Flask server
├── firms_service.py                   # NASA FIRMS VIIRS ingestion & spatial filtering
│
├── src/                               # Authoritative Major Project research source code
│   ├── data/                          # Dataset builder, DEM terrain & environmental features
│   │   ├── dataset_builder.py
│   │   ├── environmental.py
│   │   └── terrain.py
│   ├── events/                        # Spatiotemporal DBSCAN & event persistence tracking
│   │   └── event_clustering.py
│   ├── models/                        # Baseline models, BiGRU temporal net, ablations, horizons
│   │   ├── baselines.py
│   │   ├── multimodal_deep.py
│   │   ├── ablation_study.py
│   │   └── multi_horizon_eval.py
│   ├── evaluation/                    # Calibration, metrics, bootstrap CIs, LOGRO, figures
│   │   ├── calibration.py
│   │   ├── metrics.py
│   │   ├── statistical_testing.py
│   │   ├── geographic_eval.py
│   │   └── generate_figures.py
│   └── replay/                        # Multi-date retrospective replay benchmark engine
│       └── historical_replay.py
│
├── tests/                             # Automated verification test suite (52 tests)
│   ├── test_application_api.py
│   ├── test_boundary_and_geometry.py
│   ├── test_dataset_schema.py
│   ├── test_firms_service.py
│   ├── test_model_inference.py
│   └── test_wildfire_research.py
│
├── data/                              # Data directories (raw/features/events gitignored)
│   └── processed/                     # Lightweight reproducibility artifacts
│       ├── india_boundary.geojson     # Survey of India sovereign vector boundary
│       ├── india_fire_weather_final.csv # Canonical 31-feature baseline dataset
│       └── india_srtm_dem_01deg.csv   # Authoritative NOAA ETOPO DEM grid derivatives
│
├── results/                           # Authoritative experimental output artifacts
│   ├── baselines/                     # 2x2 Factorial, calibration, bootstrap CIs, PR@k
│   ├── ablations/                     # Modality block ablation comparison & metrics
│   ├── geographic/                    # LOGRO spatial cross-validation metrics & summary
│   ├── multimodal/                    # BiGRU deep net predictions & uncertainty stats
│   ├── multi_horizon/                 # T+24h, T+48h, and event persistence comparisons
│   ├── replay/                        # 20-date historical replay benchmark results
│   ├── final_model/                   # Frozen HGB baseline checkpoint & feature importance
│   └── figures/                       # Publication-grade figures (Fig 1 to Fig 4)
│
├── reports/                           # Literature matrix & external research benchmarks
│   └── literature_matrix.csv
│
├── docs/                              # Detailed scientific methodology documentation
│   ├── DATA_SOURCES.md
│   ├── DATA_SCHEMA.md
│   ├── LEAKAGE_POLICY.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── EXPERIMENT_MATRIX.md
│   ├── RESEARCH_QUESTIONS.md
│   ├── RESULTS.md
│   ├── LITERATURE_GAP.md
│   ├── MAJOR_PROJECT_ARCHITECTURE.md
│   ├── CURRENT_STATE_AUDIT.md
│   └── DEPLOYMENT.md
│
└── templates/                         # Web GIS demonstration interface
    └── index.html
```
