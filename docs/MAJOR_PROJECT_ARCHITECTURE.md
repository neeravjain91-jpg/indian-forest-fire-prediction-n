# Major Project Research Architecture

## 1. System Philosophy & Decoupling from Mini-Project

The Major Project transitions from a static 31-feature occurrence classifier into an **Event-Centric, Multimodal Spatiotemporal Wildfire Intelligence System for Sovereign India**.

While the mini-project baseline addressed *"Can we classify fire occurrence given today's meteorology?"*, the Major Project investigates:
1. **Forward Multi-Horizon Forecasting ($T+24\text{h}$, $T+48\text{h}$)**: Predicting fire risk with strictly causal temporal availability.
2. **Spatiotemporal Event Clusters**: Constructing connected fire complexes ($\text{DBSCAN-ST}$) with tracked duration, perimeters, and centroid trajectories.
3. **Multimodal Environmental Fusion**: Unifying multi-timescale atmospheric drying (ERA5-Land 1d, 3d, 7d), terrain geomorphology (elevation, slope, ruggedness), atmospheric fuel dryness (VPD, soil moisture draw-down), and antecedent fire persistence.
4. **Geographic Generalization**: Measuring spatial transfer across distinct predefined geographic fire regimes (e.g., Central Deciduous vs. Western Ghats vs. Northeast).
5. **Probability Calibration & Epistemic Uncertainty**: Brier score, Expected Calibration Error (ECE), and out-of-distribution (OOD) distance tracking.
6. **Historical Replay & Spatial Verification**: Retrospective simulation station comparing prospective forecasts against ground-truth satellite observations.

---

## 2. End-to-End Pipeline Diagram

```
                                  DATA SOURCES
   ┌────────────────────────┬────────────────────────┬────────────────────────┐
   │  NASA FIRMS VIIRS 375m │  Copernicus ERA5-Land  │   NOAA ETOPO 2022 DEM  │
   │  (SNPP, NOAA20, NOAA21)│  (Hourly Reanalysis)   │   (incorporating SRTM) │
   └───────────┬────────────┴───────────┬────────────┴───────────┬────────────┘
               │                        │                        │
               ▼                        ▼                        ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                     SPATIOTEMPORAL NORMALIZATION GRID                      │
 │                     0.1° x 0.1° Regular Analysis Grid (~11 km)             │
 └──────────────────────────────────────┬─────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         ▼                                                             ▼
 ┌──────────────────────────────┐              ┌──────────────────────────────┐
 │   EVENT-CENTRIC TRACKING     │              │    MULTIMODAL FEATURE SPACE  │
 │ - Spatiotemporal Clustering  │              │ - 1d, 3d, 7d Weather Windows │
 │ - Connected Components       │              │ - Vapor Pressure Deficit     │
 │ - Centroid Trajectories      │              │ - Soil Drought Index         │
 │ - Event Duration & Dynamics  │              │ - Terrain Geomorphology      │
 └──────────────┬───────────────┘              │ - Antecedent Fire History    │
                │                              └──────────────┬───────────────┘
                └───────────────────────┬─────────────────────┘
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                     STRICT CAUSAL LEAKAGE CONTROL                          │
 │ - Features strictly at or prior to origin T                                │
 │ - Forward Targets: T (Diagnostic), T+24h (Next-Day), T+48h, Persistence    │
 └──────────────────────────────────────┬─────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         ▼                                                             ▼
 ┌──────────────────────────────┐              ┌──────────────────────────────┐
 │    BENCHMARK BASELINE SUITE  │              │   MULTIMODAL DEEP NETWORK    │
 │ - Logistic Regression        │              │ - Weather Encoder Branch     │
 │ - Random Forest              │              │ - Environment/Terrain Branch │
 │ - HistGradientBoosting (Mini)│              │ - Fire History Branch        │
 │ - LightGBM (Primary Tree)    │              │ - Gated Cross-Modality Fusion│
 │ - Multi-Layer Perceptron     │              │ - Multi-Task Prediction Heads│
 └──────────────┬───────────────┘              │ - MC Dropout Epistemic Std   │
                │                              └──────────────┬───────────────┘
                └───────────────────────┬─────────────────────┘
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                 RIGOROUS DUAL GENERALIZATION BENCHMARKS                    │
 │ 1. Chronological Test (Train: 2018-2022, Val: 2023, Test: 2024-2025)       │
 │ 2. Spatially Disjoint Holdout (Train: Non-Central Regimes, Test: Central)   │
 │ 3. Modality Ablation Matrix & Probability Calibration (ECE, Brier)         │
 └──────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                  INTELLIGENCE PLATFORM & HISTORICAL REPLAY                 │
 │ - Real-time NASA FIRMS India Live Surveillance                             │
 │ - Calibrated Multi-Horizon Forward Risk Inference                          │
 │ - Prospective Historical Replay & Spatial Hit/Miss Verification            │
 └────────────────────────────────────────────────────────────────────────────┘
```
