# Reproducible Experiment Matrix

| Exp ID | Module / Task | Architecture / Model | Feature Space | Split Protocol | Target Variable | Artifact Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-BASE-01** | Linear Baseline | Logistic Regression + StandardScaler | 39 Multimodal | Chronological (Train 2018-22, Val 2023, Test 2024-25) | `fire` ($T$) | `results/baselines/Logistic_Regression.joblib` |
| **EXP-BASE-02** | Bagged Ensemble | Random Forest (100 trees, depth 16) | 39 Multimodal | Chronological | `fire` ($T$) | `results/baselines/Random_Forest.joblib` |
| **EXP-BASE-03** | Mini Baseline | HistGradientBoosting (lr=0.05, max_iter=300) | 39 Multimodal | Chronological | `fire` ($T$) | `results/baselines/Hist_Gradient_Boosting.joblib` |
| **EXP-BASE-04** | Gradient Boost | LightGBM (300 trees, num_leaves=31) | 39 Multimodal | Chronological | `fire` ($T$) | `results/baselines/LightGBM.joblib` |
| **EXP-BASE-05** | Neural Baseline | MLP (128x64 layers, early stopping) | 39 Multimodal | Chronological | `fire` ($T$) | `results/baselines/MLP_Neural_Baseline.joblib` |
| **EXP-DEEP-01** | Multimodal Deep | MultimodalFireNet (3 branches + Gated Fusion) | Modality Tensors (Weather, Env, History) | Chronological | Multi-task (`fire`, `lead_24h`, `persistence`) | `results/multimodal/multimodal_best_weights.pt` |
| **EXP-HORIZ-01**| Multi-Horizon | LightGBM Classifier | 39 Multimodal | Chronological | `fire` ($T$) | `results/multi_horizon/multi_horizon_comparison.csv` |
| **EXP-HORIZ-02**| Multi-Horizon | LightGBM Classifier | 39 Multimodal | Chronological | `target_fire_lead_24h` ($T+24\text{h}$) | `results/multi_horizon/multi_horizon_comparison.csv` |
| **EXP-HORIZ-03**| Multi-Horizon | LightGBM Classifier | 39 Multimodal | Chronological | `target_fire_lead_48h` ($T+48\text{h}$) | `results/multi_horizon/multi_horizon_comparison.csv` |
| **EXP-HORIZ-04**| Multi-Horizon | LightGBM Classifier | 39 Multimodal | Chronological | `target_event_persistence` | `results/multi_horizon/multi_horizon_comparison.csv` |
| **EXP-GEO-01**  | Spatial Holdout| LightGBM Multimodal | 39 Multimodal | Non-Central Train $\to$ Central Test | `fire` | `results/geographic/geographic_holdout_metrics.csv` |
| **EXP-GEO-02**  | Spatial Holdout| Random Forest Multimodal | 39 Multimodal | Non-Central Train $\to$ Central Test | `fire` | `results/geographic/geographic_holdout_metrics.csv` |
| **EXP-GEO-03**  | Spatial Holdout| HistGradientBoosting (Mini) | 31 Baseline | Non-Central Train $\to$ Central Test | `fire` | `results/geographic/geographic_holdout_metrics.csv` |
| **EXP-GEO-04**  | Spatial Holdout| Logistic Regression Baseline | 31 Baseline | Non-Central Train $\to$ Central Test | `fire` | `results/geographic/geographic_holdout_metrics.csv` |
| **EXP-ABL-01**  | Modality Ablation | LightGBM Classifier | 6 Features (Weather 1d) | Chronological | `fire` | `results/ablations/ablation_comparison.csv` |
| **EXP-ABL-02**  | Modality Ablation | LightGBM Classifier | 26 Features (Weather 1d+3d+7d) | Chronological | `fire` | `results/ablations/ablation_comparison.csv` |
| **EXP-ABL-03**  | Modality Ablation | LightGBM Classifier | 28 Features (+ Fire History) | Chronological | `fire` | `results/ablations/ablation_comparison.csv` |
| **EXP-ABL-04**  | Modality Ablation | LightGBM Classifier | 32 Features (+ Terrain & VPD) | Chronological | `fire` | `results/ablations/ablation_comparison.csv` |
| **EXP-ABL-05**  | Modality Ablation | LightGBM Classifier | 39 Features (Full Multimodal) | Chronological | `fire` | `results/ablations/ablation_comparison.csv` |
| **EXP-ABL-06**  | Modality Ablation | LightGBM Classifier (Uncalibrated) | 39 Features | Chronological | `fire` | `results/ablations/ablation_comparison.csv` |
| **EXP-REPLAY-01**| Historical Replay| HistoricalReplayEngine | 39 Multimodal | Retrospective Replay | Spatial Hit/Miss | `results/replay_demo.json` |
