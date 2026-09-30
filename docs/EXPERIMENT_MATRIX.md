# Corrected Reproducible Experiment Matrix

| Exp ID | Module / Task | Architecture / Model | Feature Space | Split Protocol | Target Variable | Artifact Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-A** | Controlled Factorial | HistGradientBoosting | 31 Baseline | Chronological (Train 18-22, Val 23, Test 24-25) | `fire` ($T$) | `results/baselines/ExpA_HGB_31_Baseline.joblib` |
| **EXP-B** | Controlled Factorial | HistGradientBoosting | 39 Multimodal | Chronological | `fire` ($T$) | `results/baselines/ExpB_HGB_39_Multimodal.joblib` |
| **EXP-C** | Controlled Factorial | LightGBM | 31 Baseline | Chronological | `fire` ($T$) | `results/baselines/ExpC_LGBM_31_Baseline.joblib` |
| **EXP-D** | Controlled Factorial | LightGBM | 39 Multimodal | Chronological | `fire` ($T$) | `results/baselines/ExpD_LGBM_39_Multimodal.joblib` |
| **EXP-BOOT-01** | Statistical Testing | 1,000 Bootstrap Resamples | 31 vs 39 Features | Test Set ($N=31,525$) | Metric $\Delta$ | `results/baselines/bootstrap_confidence_intervals.csv` |
| **EXP-LOEO-01** | Geographic Holdout | LOEO Cross-Validation (6 Folds) | 31 vs 39 Features | 5 Biomes Train $\to$ 1 Biome Test | `fire` | `results/geographic/loeo_geographic_metrics.csv` |
| **EXP-DEEP-01** | Temporal BiGRU | SpatiotemporalMultimodalFireNet | Ordered Sequence + Real DEM | Chronological | Multi-Task (`fire`, `lead`, `persistence`) | `results/multimodal/temporal_multimodal_best_weights.pt` |
| **EXP-HORIZ-01**| Multi-Horizon | LightGBM Multimodal | 39 Multimodal | Chronological | $T$, $T+24\text{h}$, $T+48\text{h}$, Persistence | `results/multi_horizon/multi_horizon_comparison.csv` |
| **EXP-ABL-01** | Modality Ablation | LightGBM | 6, 26, 28, 32, 39 Features | Chronological | `fire` | `results/ablations/ablation_comparison.csv` |
| **EXP-REPLAY-01**| Historical Replay | HistoricalReplayEngine | 39 Multimodal | 20 Test Origin Dates | $T+24\text{h}$ Verification | `results/replay/multi_date_historical_benchmark.csv` |
