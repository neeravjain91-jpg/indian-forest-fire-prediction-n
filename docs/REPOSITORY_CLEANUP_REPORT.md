# Repository Cleanup, Consolidation & Research Integrity Delivery Report

## 1. Executive Summary

This report documents the repository cleanup, structural consolidation, and research integrity pass performed on **`neeravjain91-jpg/major-forest-fire`** (grounded in validated research commit `2933ccf`).

The objective of this phase was to eliminate structural clutter, legacy duplicates carried over from the mini-project baseline, stale intermediate results, and conflicting terminology, while strictly preserving the major project's scientifically defensible contributions:
- Multimodal data representation (39 features across coordinates, multi-timescale meteorology, fuel dryness, authoritative DEM terrain, and strictly causal fire history).
- Authoritative DEM geomorphology from the NOAA ETOPO 2022 Global Relief Model with canonical Horn (1981) slope and Riley et al. (1999) Topographic Ruggedness Index.
- Controlled 2x2 factorial interaction analysis with 1,000 paired bootstrap 95% confidence intervals.
- Leave-One-Geographic-Regime-Out (LOGRO) spatial cross-validation across six predefined geographic fire regimes.
- Multi-horizon forward forecasting ($T$, $T+24\text{h}$, $T+48\text{h}$, and connected-component event persistence).
- Dual-domain historical replay evaluation reporting both candidate-domain recall ($66.17\%$) and nationwide full spatial domain recall ($3.69\%$).
- Clear separation between the baseline mini-project and the major research extension.

---

## 2. Inventory of Removed Artifacts

### A. Legacy Mini-Project Root Scripts (7 files removed)
The following scripts from the initial baseline prototype were removed because their functionality is authoritatively implemented in the `src/` hierarchy:
- `build_dataset.py` (superseded by `src/data/dataset_builder.py`)
- `download_era5_land.py` (superseded by documented Copernicus CDS API pipelines)
- `download_era5_land_daily_monthly.py` (obsolete pilot script)
- `download_era5_land_point_pilot.py` (obsolete pilot script)
- `prepare_fire_cells.py` (superseded by `src/data/dataset_builder.py`)
- `train_and_evaluate.py` (superseded by `src/models/baselines.py`)
- `train_final_model.py` (superseded by `src/models/baselines.py`)

### B. Duplicate Mini-Project Research Directory (8 files removed)
The `research/` directory was an exact duplicate of the mini-project's research track:
- `research/README.md`
- `research/ablation_plan.md`
- `research/evaluate_baselines_and_ablations.py`
- `research/experimental_protocol.md`
- `research/manuscript_outline.md`
- `research/model_training.py`
- `research/novelty_statement.md`
- `research/related_work_matrix.md`

All major research pipelines now reside strictly in `src/`, with documentation centralized in `docs/` and literature tracking in `reports/`.

### C. Stale & Superseded Experiment Result Artifacts (6 files removed)
- `results/baselines_and_ablations.json` (early mini baseline run; authoritative metrics reside in `results/baselines/baseline_comparison_metrics.csv` and `results/ablations/ablation_comparison.csv`)
- `results/replay_demo.json` (early single-date prototype; authoritative evaluation is in `results/replay/multi_date_historical_benchmark.csv` and `results/replay/multi_date_benchmark_summary.json`)
- `results/geographic/geographic_holdout_metrics.csv` and `geographic_holdout_metrics.json` (early 1-region holdout; authoritative 6-regime benchmark is in `results/geographic/loeo_geographic_metrics.csv` and `loeo_aggregate_summary.csv`)
- `results/multimodal/multimodal_metrics.json` and `multimodal_test_predictions.csv` (early tabular MLP run; authoritative BiGRU multi-scale temporal encoder output is in `results/multimodal/temporal_multimodal_metrics.json` and `temporal_multimodal_test_predictions.csv`)

### D. Superseded Figures (3 files removed)
- `results/figures/fig2_calibration_curves.png` (superseded by `fig2_reliability_diagrams.png` showing Raw vs. Platt vs. Isotonic)
- `results/figures/fig3_ablation_matrix.png` (superseded by `fig3_bootstrap_confidence_intervals.png` showing 1,000-fold bootstrap forest plot)
- `results/figures/fig4_event_distributions.png` (superseded by `fig4_loeo_geographic_spread.png` showing LOGRO cross-validation)

### E. Scratch & Temporary Files
- `prompt_589.txt`, `prompt_719.txt`, `prompt_extracted.txt` (local prompt dumps removed)

---

## 3. Retained & Authoritative Components

| Component | Path / Location | Justification & Role |
| :--- | :--- | :--- |
| **National Boundary** | `data/processed/india_boundary.geojson` | Official Survey of India sovereign vector polygon (193 KB). |
| **Meteorological Dataset** | `data/processed/india_fire_weather_final.csv` | 131,000 balanced observations (2018–2025) providing meteorological backbone. |
| **DEM Topography Grid** | `data/processed/india_srtm_dem_01deg.csv` | Authoritative NOAA ETOPO 2022 DEM elevation, Horn slope, and Riley TRI across 93,611 grid cells (~3.6 MB). |
| **Baseline Checkpoint** | `results/final_model/final_hgb_model.joblib` | Frozen 31-feature `HistGradientBoostingClassifier` verified in unit tests. |
| **Baseline Test Data** | `results/final_model/metrics.json`, `test_predictions.csv`, `feature_importance.csv` | Frozen evaluation artifacts for baseline comparisons. |
| **Major Factorial Results** | `results/baselines/` | Authoritative CSV/JSON files: `baseline_comparison_metrics.csv`, `calibration_comparison.csv`, `factorial_interaction_analysis.csv`, `bootstrap_confidence_intervals.csv`, `precision_recall_at_k.csv`. |
| **Modality Ablations** | `results/ablations/` | Authoritative CSV/JSON files: `ablation_comparison.csv`, `ablation_metrics.json`. |
| **LOGRO Spatial Results** | `results/geographic/` | Authoritative CSV files: `loeo_geographic_metrics.csv`, `loeo_aggregate_summary.csv`. |
| **Temporal BiGRU Net** | `results/multimodal/` | Authoritative files: `temporal_multimodal_metrics.json`, `temporal_multimodal_test_predictions.csv`, `uncertainty_error_correlation.csv`, `uncertainty_quantile_validation.csv`. |
| **Multi-Horizon Results** | `results/multi_horizon/` | Authoritative CSV/JSON files: `multi_horizon_comparison.csv`, `multi_horizon_metrics.json`, `multi_horizon_precision_at_k.csv`. |
| **Historical Replay** | `results/replay/` | Authoritative 20-date benchmark: `multi_date_historical_benchmark.csv`, `multi_date_benchmark_summary.json`. |
| **Authoritative Figures** | `results/figures/` | `fig1_roc_pr_curves.png`, `fig2_reliability_diagrams.png`, `fig3_bootstrap_confidence_intervals.png`, `fig4_loeo_geographic_spread.png`. |
| **External Literature** | `reports/literature_matrix.csv` | Comprehensive benchmarking matrix against 2018–2026 published literature. |

---

## 4. Documentation & Terminology Corrections

1. **Geographic Regime Terminology**:
   - Replaced all informal references to "ecoregions" or "official biomes" with **"six predefined geographic fire regimes"** across `src/data/dataset_builder.py`, `src/evaluation/generate_figures.py`, `src/evaluation/geographic_eval.py`, and all markdown documents.
   - Preserved explicit methodological disclosure noting that these boundaries are latitudinal-longitudinal macro-climatic partitions, not official WWF Terrestrial Ecoregion polygons.
2. **DEM Data Provenance**:
   - Formally standardized across all documentation to: **NOAA ETOPO 2022 Global Relief Model (v1, 15 arc-second grid)**, which natively incorporates NASA SRTM v3.0 land elevation between $60^\circ\text{N}$ and $56^\circ\text{S}$.
3. **Application & UI Status**:
   - Explicitly labeled `application.py` and `templates/index.html` as a **"Scientific Research Demonstration"**.
   - Added user-facing disclaimers clarifying that the application is an academic methodology evaluation tool and **not an operational emergency warning, civil defence alert, or disaster dispatch system**.
   - Removed unused imports and strengthened model fallback logic in `application.py` and `src/replay/historical_replay.py`.
4. **Generated Data Policy & Git Hygiene**:
   - Updated `.gitignore` to prevent repository bloat from multi-gigabyte raster archives (`*.zip`, `*.tar`, `*.tif`, `*.grib`), working manuscripts (`*.docx`, `*.pdf`), and temporary pilot benchmarks (`fire_weather_features_*.csv`).
5. **Project Configuration**:
   - Added standard open-source MIT `LICENSE` with copyright to Neerav Jain.
   - Updated `pyproject.toml` with project name `major-forest-fire` and full dependencies matching `requirements.txt`.
   - Rewrote `README.md` into 10 structured sections without hyperbolic claims.

---

## 5. Verification & Test Suite Execution

The entire automated test suite was executed across all test modules:

```bash
python -m pytest -v
```

### Execution Results:
- **`tests/test_application_api.py`**: 12 passed
- **`tests/test_boundary_and_geometry.py`**: 4 passed
- **`tests/test_dataset_schema.py`**: 7 passed
- **`tests/test_firms_service.py`**: 7 passed
- **`tests/test_model_inference.py`**: 7 passed
- **`tests/test_wildfire_research.py`**: 15 passed
- **Total Test Count**: **52 passed, 0 failed, 0 errors**
- **Test Runtime**: **5.30 seconds**
- **Pass Rate**: **100%**

All core modules were additionally validated for clean import execution:
- `src.data.dataset_builder`
- `src.models.baselines`
- `src.models.multimodal_deep`
- `src.evaluation.geographic_eval`
- `src.models.multi_horizon_eval`
- `src.models.ablation_study`
- `src.replay.historical_replay`
- `src.evaluation.generate_figures`

Zero hardcoded Windows paths or personal machine links exist in the source or documentation.

---

## 6. Final Repository Structure

```
major-forest-fire/
├── README.md                          # Concise project entry point & research summary
├── LICENSE                            # MIT Open-Source License
├── requirements.txt                   # Frozen Python environment dependencies
├── pyproject.toml                     # Project metadata & pytest configuration
├── application.py                     # Research demonstration Flask server
├── firms_service.py                   # NASA FIRMS VIIRS ingestion & spatial filtering
├── app.py                             # WSGI entrypoint wrapper
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
│   ├── conftest.py                    # Shared test fixtures & environment paths
│   ├── test_application_api.py        # Flask API endpoint verification
│   ├── test_boundary_and_geometry.py  # Spatial boundary polygon filtering tests
│   ├── test_dataset_schema.py         # Data schema & class balance tests
│   ├── test_firms_service.py          # NASA FIRMS service & security tests
│   ├── test_model_inference.py        # Baseline model loading & numerical reproducibility
│   └── test_wildfire_research.py      # Core research mathematics & edge-case unit tests
│
├── data/                              # Data directories (raw/features/events gitignored)
│   └── processed/                     # Lightweight reproducibility artifacts
│       ├── india_boundary.geojson     # Survey of India sovereign vector boundary
│       ├── india_fire_weather_final.csv # Canonical 31-feature baseline dataset
│       └── india_srtm_dem_01deg.csv   # Authoritative NOAA ETOPO DEM grid derivatives
│
├── results/                           # Authoritative experimental output artifacts
│   ├── baselines/                     # 2x2 Factorial, calibration, bootstrap CIs, PR@k
│   │   ├── baseline_comparison_metrics.csv
│   │   ├── baseline_metrics.json
│   │   ├── baseline_test_predictions.csv
│   │   ├── bootstrap_confidence_intervals.csv
│   │   ├── calibration_comparison.csv
│   │   ├── factorial_interaction_analysis.csv
│   │   └── precision_recall_at_k.csv
│   ├── ablations/                     # Modality block ablation comparison & metrics
│   │   ├── ablation_comparison.csv
│   │   └── ablation_metrics.json
│   ├── geographic/                    # LOGRO spatial cross-validation metrics & summary
│   │   ├── loeo_aggregate_summary.csv
│   │   └── loeo_geographic_metrics.csv
│   ├── multimodal/                    # BiGRU deep net predictions & uncertainty stats
│   │   ├── temporal_multimodal_metrics.json
│   │   ├── temporal_multimodal_test_predictions.csv
│   │   ├── uncertainty_error_correlation.csv
│   │   └── uncertainty_quantile_validation.csv
│   ├── multi_horizon/                 # T+24h, T+48h, and event persistence comparisons
│   │   ├── multi_horizon_comparison.csv
│   │   ├── multi_horizon_metrics.json
│   │   └── multi_horizon_precision_at_k.csv
│   ├── replay/                        # 20-date historical replay benchmark results
│   │   ├── multi_date_benchmark_summary.json
│   │   └── multi_date_historical_benchmark.csv
│   ├── final_model/                   # Frozen HGB baseline checkpoint & feature importance
│   │   ├── feature_importance.csv
│   │   ├── final_hgb_model.joblib
│   │   ├── metrics.json
│   │   └── test_predictions.csv
│   └── figures/                       # Publication-grade figures (Fig 1 to Fig 4)
│       ├── fig1_roc_pr_curves.png
│       ├── fig2_reliability_diagrams.png
│       ├── fig3_bootstrap_confidence_intervals.png
│       └── fig4_loeo_geographic_spread.png
│
├── reports/                           # Literature matrix & external research benchmarks
│   └── literature_matrix.csv
│
├── docs/                              # Detailed scientific methodology documentation
│   ├── DATA_SOURCES.md                # Authoritative data providers, resolution, licensing
│   ├── DATA_SCHEMA.md                 # Complete 39-feature dictionary & target definitions
│   ├── LEAKAGE_POLICY.md              # Temporal causality & spatial containment rules
│   ├── EVALUATION_PROTOCOL.md         # Chronological splits, LOGRO, paired bootstrap math
│   ├── EXPERIMENT_MATRIX.md           # Registry of all empirical experiments
│   ├── RESEARCH_QUESTIONS.md          # Formal hypotheses H1-H4 & verification status
│   ├── RESULTS.md                     # Comprehensive empirical results & factorial tables
│   ├── LITERATURE_GAP.md              # Comprehensive survey of wildfire literature
│   ├── MAJOR_PROJECT_ARCHITECTURE.md  # End-to-end research architecture & design
│   ├── CURRENT_STATE_AUDIT.md         # Architectural transition audit from mini baseline
│   ├── DEPLOYMENT.md                  # Operational deployment, API contracts, security
│   └── REPOSITORY_CLEANUP_REPORT.md   # This delivery report
│
└── templates/                         # Web GIS demonstration interface
    └── index.html
```

---

## 7. Known Research Limitations

1. **Retrospective Case-Control Sampling Structure**:
   - The primary dataset consists of 131,000 observations balanced 1:1 between active fire pixels and unburned grid cells at reference time $T$.
   - Output probabilities from models trained on this sample reflect retrospective sample odds rather than unconditional real-world incidence (which is $< 0.05\%$ per cell-day across sovereign India).
2. **Defensible Negative Target Labels**:
   - Target label $Y=0$ indicates the absence of confirmed VIIRS satellite thermal detections during polar overpasses.
   - Orbital gaps, cloud cover, dense smoke plumes, and sub-pixel fire sizes ($< 375\text{ m}$) mean negative labels denote observational absence, not guaranteed physical absence of ground combustion.
3. **Forward Cell Ignition Predictability**:
   - Predicting new grid-cell ignitions 24–48 hours ahead from synoptic weather and static terrain alone achieves modest discrimination (ROC-AUC $53.13\%$, PR-AUC $2.57\%$). Without real-time human activity monitoring or lightning telemetry, arbitrary ignition forecasting is ill-posed.
   - Conversely, connected-component event persistence is well-posed ($68.39\%$ ROC-AUC, $18.0\%$ Top-100 Precision).
4. **Historical Replay Dual-Domain Interpretation**:
   - Observed historical replay recall of $66.17\%$ is strictly conditioned on the monitored candidate domain active at time $T$.
   - Nationwide unconditioned spatial recall is $3.69\%$, illustrating that prospective surveillance tracking active fire corridors cannot replace nationwide new ignition detection.
5. **Non-Operational Academic Status**:
   - The web interface is strictly an academic research demonstration and verification platform. It is not designed or certified for emergency disaster dispatch, civil defence warning, or automated wildfire response.
