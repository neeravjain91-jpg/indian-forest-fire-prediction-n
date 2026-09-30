# Multimodal Data Schema & Feature Hierarchy

## 1. Feature Hierarchy & Modalities

The Major Project organizes predictors into six distinct modalities:

```
Multimodal Input Feature Space (39 Total Features)
├── 1. Spatiotemporal Coordinates (5 features)
│   ├── grid_lat [float]
│   ├── grid_lon [float]
│   ├── hour [int]
│   ├── year [int]
│   └── month [int]
├── 2. Weather State & Temporal History (26 features)
│   ├── 1-Day Weather Window (6 features: temp_1d, rh_1d, wind_1d, pressure_1d, soil_1d, rain_1d)
│   ├── 3-Day Weather Window (10 features: temp_3d_mean/max/min, rh_3d_mean/min, wind_3d_mean/max, pressure_3d_mean, soil_3d_mean, rain_3d_total)
│   └── 7-Day Weather Window (10 features: temp_7d_mean/max/min, rh_7d_mean/min, wind_7d_mean/max, pressure_7d_mean, soil_7d_mean, rain_7d_total)
├── 3. Environmental / Fuel Dryness Dynamics (2 features)
│   ├── vpd_1d [float] (Vapor Pressure Deficit at 1d, kPa)
│   └── vpd_3d_mean [float] (Mean VPD over 3d, kPa)
├── 4. Topography & Terrain Geomorphology (3 features)
│   ├── elevation_m [float] (SRTM-derived mean elevation, m)
│   ├── slope_deg [float] (SRTM-derived terrain slope, degrees)
│   └── ruggedness_index [float] (Topographic ruggedness index)
└── 5. Antecedent Fire History & Spatiotemporal Persistence (3 features)
    ├── fire_history_recurrence [float] (Cell-level historical fire recurrence rate)
    ├── antecedent_fire_24h [int] (Fire detection presence in cell during previous 24h)
    └── active_cluster_proximity_km [float] (Distance to nearest active spatiotemporal fire cluster, km)
```

---

## 2. Target Variables & Forecast Horizons

| Target Variable | Horizon | Formulation | Valid Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `fire_lead_24h` | $T+24\text{h}$ | Binary Classification | $\{0, 1\}$ | Indicates whether active fire detection occurs in the grid cell in the next 24-hour cycle. |
| `fire_lead_48h` | $T+48\text{h}$ | Binary Classification | $\{0, 1\}$ | Indicates whether active fire detection occurs in the grid cell in the $T+24\text{h}$ to $T+48\text{h}$ window. |
| `event_persistence` | $T+24\text{h}$ | Binary Classification | $\{0, 1\}$ | For active fire cells at $T$, whether the cluster continues burning into $T+24\text{h}$. |
| `event_displacement_km` | $T+24\text{h}$ | Continuous Regression | $[0, \infty)$ | Measured centroid displacement distance of active fire event over 24 hours. |

---

## 3. Strict Data Type & Missing Value Constraints

* **Completeness**: No missing (`NaN` or `null`) values permitted in final analysis features.
* **Coordinate Precision**: Latitudes and longitudes strictly standardized to $0.1^\circ$ grid centroids (rounded to 1 decimal place).
* **Timestamps**: All temporal references strictly formatted in ISO 8601 (`YYYY-MM-DD` and UTC hour).
