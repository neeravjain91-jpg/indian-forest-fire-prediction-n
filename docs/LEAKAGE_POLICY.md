# Formal Data Leakage Control & Evaluation Policy

## 1. Principle of Strict Causal Information Availability

For any prediction made at forecast reference time $T$:
$$\text{Information Set}(\mathcal{I}_T) = \{X_i(t) \mid t \le T\}$$

Under no circumstances may features include information where $t > T$.

```
TIME AXIS: ───[T - 7d]──────[T - 3d]──────[T - 1d]──────[T (Forecast Origin)] ──────[T + 24h]──────[T + 48h]───►
               ◄────────── ALLOWED PREDICTORS ──────────►                  ◄────── PREDICTION TARGETS ──────►
               - Weather 7d/3d/1d                                           - Fire Occurrence at T+24h
               - Antecedent Fire Recurrence                                  - Fire Occurrence at T+48h
               - Topography / Static Terrain                                 - Event Persistence
               - VPD & Soil Moisture Deficit                                 - Event Displacement
```

---

## 2. Permitted vs. Prohibited Features

| Category | Permitted ($t \le T$) | Strictly Prohibited ($t > T$ or Contemporaneous Target Artifacts) |
| :--- | :--- | :--- |
| **Active Fire Detections** | Historical fire counts prior to $T$; cluster persistence up to $T$. | Fire counts at $T+24\text{h}$; FRP at target horizon; brightness temp at target horizon. |
| **Meteorology** | ERA5 weather aggregated over $[T-168\text{h}, T]$, $[T-72\text{h}, T]$, $[T-24\text{h}, T]$. | Weather forecasts or actual weather occurring at or after $T$. |
| **Fire Intensity (FRP)** | Historical max FRP of past events prior to $T$. | FRP of the target fire being predicted (circular reasoning / label leakage). |
| **Vegetation / Moisture** | Soil moisture and VPD measured prior to or at $T$. | Post-ignition burn severity indices, dNBR, or post-fire vegetation drop. |

---

## 3. Spatial Leakage & Autocorrelation Mitigation

Standard random $k$-fold cross-validation or random train/test splits cause severe optimistic bias because neighboring pixels ($< 50\text{ km}$) share identical synoptic weather patterns, vegetation regimes, and lightning/human ignition probabilities.

To enforce strict spatial integrity:
1. **Chronological Splitting (Temporal Generalization)**:
   - **Training Set**: 2018–2022 (5 full calendar years).
   - **Validation Set**: 2023 (1 full calendar year).
   - **Test Set**: 2024–2025 (2 full calendar years).
   - No temporal overlap across splits.
2. **Geographically Disjoint Regional Holdout (Spatial Generalization)**:
   - India is segmented into 6 ecologically cohesive regional zones:
     - `CENTRAL`: Deccan Plateau / Central Teak & Sal dry deciduous forest.
     - `WESTERN_GHATS`: Southwestern montane & moist evergreen forest.
     - `NORTHEAST`: Subtropical & temperate Indo-Burma biodiversity hotspot.
     - `NORTH`: Himalayan foothills, Siwaliks, and subtropical pine.
     - `EAST`: Eastern Ghats, Chota Nagpur plateau, and coastal hinterland.
     - `NORTHWEST`: Aravalli range and semi-arid thorn scrub.
   - Models are trained on 5 regions and tested on a completely disjoint, unseen 6th region to measure true out-of-region generalizability.
