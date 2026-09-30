# Multimodal Data Schema & Feature Hierarchy

## 1. Feature Hierarchy & Modalities (39 Total Features)

```
Multimodal Input Feature Space (39 Total Features)
├── 1. Spatiotemporal Coordinates (5 features)
│   ├── grid_lat [float] (0.10° grid centroid latitude, degrees North)
│   ├── grid_lon [float] (0.10° grid centroid longitude, degrees East)
│   ├── hour [int] (Acquisition UTC hour, 0-23)
│   ├── year [int] (Observation year, 2018-2025)
│   └── month [int] (Observation month, 1-12)
├── 2. Atmospheric Meteorology & Multi-Scale Temporal Representation (26 features)
│   ├── 1-Day Weather Window (6 features: temp_1d, rh_1d, wind_1d, pressure_1d, soil_1d, rain_1d)
│   ├── 3-Day Weather Window (10 features: temp_3d_mean/max/min, rh_3d_mean/min, wind_3d_mean/max, pressure_3d_mean, soil_3d_mean, rain_3d_total)
│   └── 7-Day Weather Window (10 features: temp_7d_mean/max/min, rh_7d_mean/min, wind_7d_mean/max, pressure_7d_mean, soil_7d_mean, rain_7d_total)
├── 3. Environmental & Atmospheric Fuel Dryness Proxies (3 features)
│   ├── vpd_1d [float] (Instantaneous Vapor Pressure Deficit at 1d, kPa via Tetens formula)
│   ├── vpd_3d_mean [float] (3-day antecedent drying proxy evaluated on mean T and mean RH, kPa)
│   └── soil_drought_index [float] (Surface soil moisture deficit proxy relative to nominal 0.35 m^3/m^3 reference threshold, [0, 1])
├── 4. Authoritative DEM Terrain Geomorphology (3 features)
│   ├── elevation_m [float] (NOAA ETOPO 2022 Global Relief Model integrating NASA SRTM v3 elevation, meters above sea level)
│   ├── slope_deg [float] (Canonical Horn (1981) 3x3 weighted finite-difference slope gradient, degrees)
│   └── ruggedness_index [float] (Riley et al. (1999) Topographic Ruggedness Index over 8 spatial neighbors, meters)
└── 5. Strictly Causal Fire History & Persistence (2 features)
    ├── fire_history_recurrence [float] (Annualized fire detection rate in grid cell strictly prior to T, t < T)
    └── antecedent_fire_24h [int] (Binary indicator: fire detected in grid cell on calendar date T - 1 day)
```

---

## 2. Target Variables & Forward Forecast Formulations

| Target Variable | Horizon | Formulation | Valid Values | Test Positive Prevalence | Description |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `fire` | $T$ (Diagnostic) | Binary Classification | $\{0, 1\}$ | 50.00% (Balanced) | Indicates whether active VIIRS thermal anomaly was confirmed in grid cell at reference observation cycle $T$. |
| `target_fire_lead_24h` | $T+24\text{h}$ (Next-Day) | Binary Classification | $\{0, 1\}$ | 2.32% (Rare) | Indicates whether active VIIRS thermal anomaly was confirmed in grid cell on calendar date $T + 1\text{ day}$. |
| `target_fire_lead_48h` | $T+48\text{h}$ (Two-Day) | Binary Classification | $\{0, 1\}$ | 2.58% (Rare) | Indicates whether active VIIRS thermal anomaly was confirmed in grid cell on calendar date $T + 2\text{ days}$. |
| `target_event_persistence` | $T+24\text{h}$ | Binary Classification | $\{0, 1\}$ | 6.87% (Uncommon) | Indicates whether an active connected fire complex at $T$ continued burning into $T + 1\text{ day}$ within $\le 25\text{ km}$ spatial proximity. |

---

## 3. Methodological Disclosures & Satellite Observation Constraints

1. **Retrospective Case-Control Sampling vs. Population Prevalence**:
   - The 131,000-observation dataset is constructed as a 1:1 retrospective case-control sample at reference time $T$ ($P(Y=1) = 0.50$).
   - Raw model output probabilities reflect sample odds, not absolute unconditional population risk.
   - Across sovereign India's ~93,611 grid cells ($0.10^\circ \approx 11.1\text{ km}$), true daily active fire prevalence is typically $< 0.05\%$. Interpreting model scores as population risk requires prior probability odds adjustment via Bayes' rule.
2. **Defensible Negative Labels ($Y=0$)**:
   - VIIRS instruments aboard polar sun-synchronous satellites (Suomi-NPP, NOAA-20, NOAA-21) observe India approximately twice in 24 hours.
   - A negative target label ($Y=0$) denotes the *absence of a confirmed satellite thermal detection* during orbital passes. It is constrained by satellite overpass timing, cloud cover, thick smoke obscuration, and sensor detection limits ($375\text{ m}$ pixel resolution; sub-pixel fire radiative power threshold). It does not guarantee the complete physical absence of sub-canopy smoldering.
3. **Event Persistence Linkage Criterion**:
   - For an active detection at $(x, y, T)$ belonging to event complex $C_i$:
     $\text{Target}_{\text{persistence}} = 1$ if and only if cluster $C_i$ contains active satellite detections on calendar day $T+1\text{d}$ AND at least one detection is within great-circle distance $\le 25.0\text{ km}$ of $(x, y)$.
   - Detections separated by multi-day gaps, belonging to different clusters, or located $> 25\text{ km}$ away along chained clusters are assigned 0.
4. **VPD Aggregation Approximation**:
   - Because saturation vapor pressure $e_s(T)$ is strictly convex in temperature $T$, Jensen's inequality implies $\mathbb{E}[VPD(T, RH)] \ge VPD(\mathbb{E}[T], \mathbb{E}[RH])$.
   - The 3-day feature `vpd_3d_mean` is computed as $VPD(\bar{T}, \overline{RH})$ from ERA5-Land 3-day mean temperature and relative humidity. This serves as a computationally efficient, smooth multi-timescale atmospheric fuel drying proxy that mildly dampens diurnal extremes.
5. **Soil Moisture Deficit Proxy**:
   - Topsoil moisture deficit is calculated as $SDP = \max(0, 0.35 - \theta) / 0.35$, referencing a nominal national baseline of $0.35\text{ m}^3/\text{m}^3$.
   - This represents relative surface fuel desiccation rather than localized hydraulic field capacity (which varies geographically across India's soil orders).
