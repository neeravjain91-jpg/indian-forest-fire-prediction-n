# Multimodal Data Schema & Feature Hierarchy

## 1. Feature Hierarchy & Modalities (39 Total Features)

```
Multimodal Input Feature Space (39 Total Features)
├── 1. Spatiotemporal Coordinates (5 features)
│   ├── grid_lat [float] (0.1° grid centroid latitude, degrees North)
│   ├── grid_lon [float] (0.1° grid centroid longitude, degrees East)
│   ├── hour [int] (Acquisition UTC hour, 0-23)
│   ├── year [int] (Observation year, 2018-2025)
│   └── month [int] (Observation month, 1-12)
├── 2. Atmospheric Meteorology & Temporal History (26 features)
│   ├── 1-Day Weather Window (6 features: temp_1d, rh_1d, wind_1d, pressure_1d, soil_1d, rain_1d)
│   ├── 3-Day Weather Window (10 features: temp_3d_mean/max/min, rh_3d_mean/min, wind_3d_mean/max, pressure_3d_mean, soil_3d_mean, rain_3d_total)
│   └── 7-Day Weather Window (10 features: temp_7d_mean/max/min, rh_7d_mean/min, wind_7d_mean/max, pressure_7d_mean, soil_7d_mean, rain_7d_total)
├── 3. Environmental & Atmospheric Fuel Dryness (3 features)
│   ├── vpd_1d [float] (Vapor Pressure Deficit at 1d, kPa via Tetens formula)
│   ├── vpd_3d_mean [float] (Mean VPD over 3-day antecedent window, kPa)
│   └── soil_drought_index [float] (Normalized topsoil moisture draw-down index, [0, 1])
├── 4. Authoritative DEM Terrain Geomorphology (3 features)
│   ├── elevation_m [float] (Real NOAA ETOPO / SRTM elevation above sea level, meters)
│   ├── slope_deg [float] (Horn's finite-difference spatial slope gradient, degrees)
│   └── ruggedness_index [float] (Riley Topographic Ruggedness Index, meters elevation variance)
└── 5. Strictly Causal Fire History & Persistence (2 features)
    ├── fire_history_recurrence [float] (Annualized fire detection rate in cell strictly prior to T, t < T)
    └── antecedent_fire_24h [int] (Binary indicator: fire detected in cell on date T - 1 day)
```

---

## 2. Target Variables & Forward Forecast Formulations

| Target Variable | Horizon | Formulation | Valid Values | Test Positive Prevalence | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fire` | $T$ (Diagnostic) | Binary Classification | $\{0, 1\}$ | 50.00% (Balanced) | Indicates whether active VIIRS fire was detected in grid cell at reference observation cycle $T$. |
| `target_fire_lead_24h` | $T+24\text{h}$ (Next-Day) | Binary Classification | $\{0, 1\}$ | 2.32% (Rare) | Indicates whether active VIIRS fire was detected in grid cell on calendar date $T + 1\text{ day}$. |
| `target_fire_lead_48h` | $T+48\text{h}$ (Two-Day) | Binary Classification | $\{0, 1\}$ | 2.58% (Rare) | Indicates whether active VIIRS fire was detected in grid cell on calendar date $T + 2\text{ days}$. |
| `target_event_persistence` | $T+24\text{h}$ | Binary Classification | $\{0, 1\}$ | 6.87% (Uncommon) | Indicates whether an active connected fire complex at $T$ continued burning into $T + 1\text{ day}$. |

---

## 3. Satellite Observation Constraints & Negative Label Interpretation

1. **Opportunistic Polar Orbiters**: VIIRS instruments fly on polar sun-synchronous satellites (Suomi-NPP, NOAA-20, NOAA-21) passing over India approximately twice every 24 hours (around 01:30 and 13:30 local solar time). Observations are discrete snapshots, not continuous hourly streams.
2. **Negative Label Caveat**: A target value of `0` denotes the *absence of a confirmed satellite detection* during overpasses on that day. It cannot guarantee the complete physical absence of sub-canopy smoldering or small fires obscured by cloud cover or heavy smoke plumes.
3. **Imbalance & Metric Handling**: Because true forward occurrence is naturally sparse ($\sim 2.3\%$), standard classification accuracy is uninformative. Models are evaluated using PR-AUC, ROC-AUC, Brier score, ECE, and Top-$k$ Precision ($k \in \{100, 250, 500, 1000\}$).
