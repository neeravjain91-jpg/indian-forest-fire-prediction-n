# Audit of Existing Repository State & Transition Plan to Major Project

## 1. Executive Summary

This audit assesses the state of the repository prior to the Major Project transition. 
The repository previously embodied the *mini-project* baseline: a single-modality (meteorological + coordinate) static cell classification model using `HistGradientBoostingClassifier`. While methodologically sound as an empirical occurrence baseline, it does not constitute an event-centric spatiotemporal forecasting or intelligence system.

This audit establishes:
1. What was inherited from the mini-project baseline.
2. The fundamental scientific and engineering limitations of that baseline.
3. The structural, algorithmic, and data architecture required to evolve this into an authoritative **Major Research Project**.

---

## 2. Inventory of Baseline Components

| Component | Path / Location | Current Role | Major Project Transition Role |
| :--- | :--- | :--- | :--- |
| **Processed Dataset** | `data/processed/india_fire_weather_final.csv` | 131,000 balanced observations (2018–2025) at 0.1° resolution. Weather features at 1d, 3d, 7d. | Retained as meteorological backbone; augmented with event-centric attributes, antecedent fire persistence, terrain, vegetation, and lead-lag forward horizons. |
| **Boundary GeoJSON** | `data/processed/india_boundary.geojson` | Survey of India national boundary (193 KB). | Retained for spatial filtering and dashboard GIS rendering. |
| **Baseline Model** | `results/final_model/final_hgb_model.joblib` | 31-feature `HistGradientBoostingClassifier`. | Formally demoted to **Baseline 3 (HGB Baseline)** in the multi-baseline benchmark suite. |
| **Baseline Metrics** | `results/final_model/metrics.json` | Test Acc: 70.01%, ROC-AUC: 78.52%. | Benchmarked against Logistic Regression, Random Forest, LightGBM, and Proposed Multimodal Spatiotemporal Deep Model. |
| **Flask Backend** | `application.py` | Serves static model inference and FIRMS live fetch. | Refactored into a multi-mode Intelligence Platform: Live Surveillance, Multimodal Forecasting, Event Intelligence, Historical Replay, and Uncertainty/Generalization audit. |
| **Web UI** | `templates/index.html` | GIS map and 31-input prediction form. | Upgraded with Replay scrubber, Event clusters/trajectories, Multi-horizon forecasts ($T+24\text{h}$, $T+48\text{h}$), and Uncertainty overlays. |
| **FIRMS Ingestion** | `firms_service.py` | Area API wrapper with client key session setter. | Hardened: removed unsafe `/api/set-key` endpoint, enforced server-side environment variables, resilient caching, and request timeouts. |

---

## 3. Scientific & Methodological Limitations of the Baseline

1. **Synchronous Classification vs. Genuine Forward Forecasting**:
   - The baseline predicts fire detection occurrence at time $t$ using meteorological aggregations centered at time $t$ ($t-24\text{h}$ to $t$). This answers *"Was this weather conducive to fire today?"* rather than *"Will a fire ignite, persist, or spread in this location over the next 24 to 48 hours?"*
   - Major Project Requirement: Mathematically define forward forecast horizons ($T+24\text{h}$ and $T+48\text{h}$) where features at time $T$ predict fire activity at $T+\Delta t$ with strictly causal temporal gating.

2. **Point / Cell Independence vs. Event-Centric Dynamics**:
   - The baseline treats each 0.1° cell on each day as an independent, identically distributed (i.i.d.) point. Real wildfires are connected spatiotemporal events that ignite, grow, cluster, disperse smoke/heat, and persist across contiguous days and adjacent cells.
   - Major Project Requirement: Construct spatiotemporal fire clusters ($\text{DBSCAN-ST}$), tracking event duration, bounding boxes, centroid trajectories, detection counts, and expansion kinetics.

3. **Absence of Multimodal Environmental Context**:
   - The baseline relied solely on 2-meter air temperature, relative humidity, wind speed, pressure, soil moisture, and precipitation. It lacked:
     - **Terrain Geomorphology**: Elevation, slope, aspect, topographic ruggedness (which fundamentally govern fire propagation and flammability).
     - **Antecedent Fire History & Persistence**: Time-since-last-fire, local recurrence rate, proximity to active clusters.
     - **Vegetation / Fuel Dryness Dynamics**: Fuel availability and vapor pressure deficit (VPD).
   - Major Project Requirement: Build a multimodal architecture with dedicated feature encoders for Weather, Environment/Terrain, and Fire History.

4. **Evaluation Vulnerabilities & Lack of Geographic Stress-Testing**:
   - Baseline evaluation relied strictly on a single chronological split (2018–2022 train, 2023 val, 2024–2025 test). While avoiding temporal leakage, it provided zero insight into whether the model learned localized spatial memorization or generalizable physical fire regimes.
   - Major Project Requirement: Implement spatially disjoint regional/ecological holdout benchmarks across India (e.g., Central Deciduous vs. Western Ghats vs. Northeast Subtropical).

5. **Lack of Probability Calibration and Epistemic Uncertainty**:
   - Raw tree probabilities were uncalibrated, with no Brier Score, Expected Calibration Error (ECE), or Out-Of-Distribution (OOD) tracking.
   - Major Project Requirement: Implement Platt / Isotonic calibration, reliability diagrams, and epistemic uncertainty tracking to identify when model confidence degrades.

---

## 4. Architectural Transformation Plan

```
[Mini-Project Baseline]               [Major Project Research Architecture]
-----------------------               -------------------------------------
Static 0.1° Grid Occurrence    ───►   Event-Centric Spatiotemporal Forecasting (T+24h, T+48h)
31 Weather-Only Features       ───►   Multimodal Inputs: Weather + Terrain + Vegetation + Fire History
Single HGB Classifier          ───►   Multi-Baseline Suite (LR, RF, HGB, LightGBM, MLP) + Multimodal Deep Model
Single Chronological Split     ───►   Dual Evaluation: Chronological + Spatially Disjoint Regional Holdouts
Uncalibrated Tree Probability  ───►   Calibrated Predictions (ECE, Brier) + Uncertainty / OOD Indicator
Point Inference Form           ───►   Intelligence Dashboard: Live + Multi-Horizon Forecast + Replay + Event Tracks
```
