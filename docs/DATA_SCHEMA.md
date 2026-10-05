# Dataset Schema & Feature Hierarchy

## 1. Overview

The primary dataset for this research is:
`data/processed/india_fire_weather_final.csv`

- **Total Observations**: 131,000
- **Class Balance**: 65,518 fire ($Y=1$) vs. 65,482 non-fire ($Y=0$)
- **Spatial Resolution**: 0.10° grid cells (~11.1 km)
- **Unique Spatial Cells**: 26,494
- **Temporal Coverage**: 2018–2025 (8 full calendar years)
- **Missing / Null Values**: Exactly 0 across all 31 predictive features
- **Total Columns**: 38 (31 predictive features, 1 binary target, 6 metadata/telemetry columns)

---

## 2. Canonical 31 Predictive Features

The feature vector $X \in \mathbb{R}^{31}$ is structured into three clear spatiotemporal and meteorological categories:

```
Canonical 31 Feature Space
├── Spatial Coordinates (2 features)
│   ├── grid_lat [float] (0.10° grid cell centroid latitude, degrees North: 6.0° to 38.0°)
│   └── grid_lon [float] (0.10° grid cell centroid longitude, degrees East: 68.0° to 98.0°)
├── Temporal Cyclical Features (3 features)
│   ├── hour [int] (Observation acquisition UTC hour, 0–23)
│   ├── year [int] (Calendar observation year, 2018–2025)
│   └── month [int] (Calendar observation month, 1–12)
├── 1-Day Weather Window (6 features)
│   ├── temp_1d [float] (2-meter air temperature at 1-day lag, °C)
│   ├── rh_1d [float] (Relative humidity at 1-day lag, %)
│   ├── wind_1d [float] (10-meter wind speed at 1-day lag, m/s)
│   ├── pressure_1d [float] (Surface barometric pressure at 1-day lag, hPa)
│   ├── soil_1d [float] (Topsoil volumetric moisture 0–7 cm at 1-day lag, m³/m³)
│   └── rain_1d [float] (Precipitation accumulation over 1-day lag, mm)
├── 3-Day Weather Window (10 features)
│   ├── temp_3d_mean [float] (Mean 2m temperature over antecedent 72 hours, °C)
│   ├── temp_3d_max [float] (Maximum 2m temperature over antecedent 72 hours, °C)
│   ├── temp_3d_min [float] (Minimum 2m temperature over antecedent 72 hours, °C)
│   ├── rh_3d_mean [float] (Mean relative humidity over antecedent 72 hours, %)
│   ├── rh_3d_min [float] (Minimum relative humidity over antecedent 72 hours, %)
│   ├── wind_3d_mean [float] (Mean wind speed over antecedent 72 hours, m/s)
│   ├── wind_3d_max [float] (Maximum wind speed over antecedent 72 hours, m/s)
│   ├── pressure_3d_mean [float] (Mean surface pressure over antecedent 72 hours, hPa)
│   ├── soil_3d_mean [float] (Mean soil moisture over antecedent 72 hours, m³/m³)
│   └── rain_3d_total [float] (Total accumulated rainfall over antecedent 72 hours, mm)
└── 7-Day Weather Window (10 features)
    ├── temp_7d_mean [float] (Mean 2m temperature over antecedent 168 hours, °C)
    ├── temp_7d_max [float] (Maximum 2m temperature over antecedent 168 hours, °C)
    ├── temp_7d_min [float] (Minimum 2m temperature over antecedent 168 hours, °C)
    ├── rh_7d_mean [float] (Mean relative humidity over antecedent 168 hours, %)
    ├── rh_7d_min [float] (Minimum relative humidity over antecedent 168 hours, %)
    ├── wind_7d_mean [float] (Mean wind speed over antecedent 168 hours, m/s)
    ├── wind_7d_max [float] (Maximum wind speed over antecedent 168 hours, m/s)
    ├── pressure_7d_mean [float] (Mean surface pressure over antecedent 168 hours, hPa)
    ├── soil_7d_mean [float] (Mean soil moisture over antecedent 168 hours, m³/m³)
    └── rain_7d_total [float] (Total accumulated rainfall over antecedent 168 hours, mm)
```

---

## 3. Target Variable Definition

| Target Column | Type | Values | Description |
| :--- | :--- | :---: | :--- |
| `fire` | Binary | $\{0, 1\}$ | Indicates whether a confirmed VIIRS thermal anomaly occurred within the spatial 0.1° grid cell at reference acquisition cycle $T$. |

- **Positive Class ($Y=1$)**: Verified satellite active fire detection by the Suomi-NPP VIIRS 375m sensor.
- **Negative Class ($Y=0$)**: Controlled background non-detection cell sampled from within the same geographic and temporal envelope.

---

## 4. Metadata & Excluded Telemetry Columns

The dataset also contains 6 observational metadata columns that are **strictly excluded** from the predictive feature space to prevent data leakage:

| Column | Excluded Reason |
| :--- | :--- |
| `acq_date` | Used solely for chronological train/validation/test partitioning. |
| `latitude` | Raw continuous satellite observation latitude (represented by canonical `grid_lat`). |
| `longitude` | Raw continuous satellite observation longitude (represented by canonical `grid_lon`). |
| `frp` | Fire Radiative Power (MW) — direct physical signature of fire intensity; contemporaneous telemetry leakage. |
| `confidence` | Satellite algorithm confidence categorization (`nominal`, `high`) — contemporaneous telemetry leakage. |
| `satellite` | Sensor platform name (`Suomi-NPP`) — administrative tracking metadata. |

---

## 5. Methodological Considerations

1. **Case-Control Sampling Context**: The 1:1 balance in the processed dataset reflects an experimental retrospective case-control design. The model outputs estimate sample-level odds of fire occurrence under matching spatial and temporal baselines, not raw nationwide unconditional incidence.
2. **Sensor Detection Boundaries**: A non-detection ($Y=0$) denotes the absence of a confirmed satellite thermal anomaly during orbital passes. It is influenced by satellite overpass timing (sun-synchronous orbit, ~1:30 PM/AM local solar time), cloud cover, and canopy density.
