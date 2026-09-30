# Research Literature Review & Scientific Gap Analysis

## 1. Context and Problem Formulation

Machine learning applied to wildfire intelligence has accelerated over the past decade, driven by the availability of moderate-resolution satellite active-fire detections (e.g., MODIS, VIIRS) and gridded atmospheric reanalyses (e.g., ERA5, HRRR). However, rigorous critical surveys (notably **Jain et al., 2020**; **Ploton et al., 2020**; **Meyer et al., 2018**) have revealed pervasive methodological flaws in published literature:

1. **Spatial Autocorrelation and Fake Performance**: Models evaluated on standard random train/test splits inadvertently leak spatial coordinate and regional information, leading to artificially inflated accuracy metrics ($>95\%$) that collapse completely when applied to new geographic regions.
2. **Synchronous Occurrence vs. True Forward Forecasting**: The majority of papers classify whether a fire occurred on day $d$ using weather features aggregated during day $d$. This is a diagnostic or diagnostic-occurrence task, not a forward forecast.
3. **Point/Cell Atomization vs. Event Coherence**: Treating each grid cell or hotspot detection as an independent observation ignores the physical reality that wildfires are spatially contiguous, temporally persisting events with identifiable ignition points, perimeters, trajectories, and lifetimes (**Andela et al., 2019**).
4. **Neglect of Satellite Orbital Constraints**: Many works either assume continuous temporal monitoring or treat satellite observations as instantaneous truth without addressing sensor overpass cadence, cloud blockage, or detection limits.

---

## 2. Scientific Gap Analysis: Global vs. Indian Context

| Dimension | Typical Literature State (Global) | Typical Literature State (India) | Major Project Contribution |
| :--- | :--- | :--- | :--- |
| **Prediction Target** | Static susceptibility or simultaneous daily occurrence (**Cortez & Morais, 2007**; **Sharma et al., 2024**) | Static GIS susceptibility or historical frequency maps (**Reddy et al., 2017**; **Kale et al., 2017**) | **Forward Lead-Time Forecasting ($T+24\text{h}$, $T+48\text{h}$)** and event persistence under strict causal time-gating. |
| **Observation Unit** | Isolated pixel/cell or arbitrary image patch (**Radke et al., 2019**) | Isolated MODIS/FSI hotspot point counts | **Event-Centric Spatiotemporal Clusters**: Connected spatiotemporal events with tracked centroid trajectories, bounding boxes, and duration. |
| **Input Modalities** | Weather-only or imagery-only (**Huot et al., 2022**) | Coarse static GIS layers (elevation, distance to roads) | **Multimodal Integration**: Temporal weather dynamics (1d, 3d, 7d), terrain geomorphology (elevation, slope, aspect), fuel/dryness proxies (VPD, soil moisture), and antecedent fire persistence. |
| **Spatial Evaluation** | Random $k$-fold cross-validation (vulnerable to spatial autocorrelation) | Random split or single administrative boundary holdout | **Spatially Disjoint Regional & Ecological Holdout**: Training and testing in disjoint ecological zones across India. |
| **Uncertainty & Calibration** | Uncalibrated probabilities without reliability verification | Completely omitted | **Brier Score, ECE, and Out-Of-Distribution (OOD)** distance-to-support tracking. |

---

## 3. The Central Research Question and Hypotheses

### Central Research Question
> *Can an event-centric, multimodal spatiotemporal framework combining multi-timescale meteorological history, antecedent fire persistence, terrain, and fuel dryness produce superior and more transferable forward wildfire forecasts ($T+24\text{h}$, $T+48\text{h}$) across India than static, location-dependent occurrence classifiers?*

### Formal Hypotheses
* **H1 (Multi-timescale weather dynamics)**: Integrating antecedent multi-day atmospheric drying signals (1d, 3d, 7d cumulative drying, VPD, soil moisture draw-down) yields statistically superior forward fire discrimination compared to instantaneous weather alone.
* **H2 (Event-centric representation)**: Explicitly encoding spatiotemporal event cluster characteristics (cluster age, active neighbor density, distance to active front) improves forward persistence and occurrence prediction over treating cells as isolated points.
* **H3 (Multimodal geographic transferability)**: A multimodal architecture incorporating terrain geomorphology and environmental proxies reduces the performance gap when tested on geographically held-out ecological regions compared to location-heavy baseline models.
* **H4 (Calibration & uncertainty tracking)**: Post-hoc probability calibration and epistemic/OOD uncertainty scoring reliably identify instances where forward forecast reliability degrades in unseen climatic regimes.

---

## 4. Defensible Scope & Boundary of Claims

In accordance with strict scientific integrity:
1. **No Claims of Physical CFD Fire-Spread Simulation**: Our empirical movement tracking models statistical centroid displacement and cluster expansion; it does not solve physical Navier-Stokes combustion equations.
2. **Defensible Forecast Horizons**: Because VIIRS active-fire telemetry over India is acquired via polar-orbiting satellites (~2 daytime and nighttime passes per 24 hours per sensor), continuous hourly targets ($T+1\text{h}$, $T+2\text{h}$) are scientifically unsupportable without continuous geostationary fire products. Therefore, **$T+24\text{h}$ (Next-Day)** and **$T+48\text{h}$ (Two-Day)** represent the highest defensible forecast horizons.
3. **No Operational Claim**: The framework is a scientific research benchmark, not a certified emergency early-warning system.
