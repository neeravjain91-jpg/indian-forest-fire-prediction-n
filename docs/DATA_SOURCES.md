# Data Sources & Acquisition Protocols

## 1. Primary Data Sources

This research relies on two authoritative open Earth observation datasets, integrated across sovereign India from 2018 to 2025:

### 1.1 NASA FIRMS Active Fire Observations
- **Sensor**: Visible Infrared Imaging Radiometer Suite (VIIRS) aboard the Suomi National Polar-orbiting Partnership (Suomi-NPP) satellite.
- **Data Product**: VNP14IMGTDL (NRT and archive active fire product).
- **Spatial Resolution**: 375-meter nominal ground resolution at nadir (I-Band I4, 3.55–3.93 µm).
- **Temporal Coverage**: January 1, 2018 to December 31, 2025.
- **Geographic Extent**: Sovereign India (bounding box roughly 6°N–38°N, 68°E–98°E).
- **Key Attributes**: Latitude, Longitude, Acquisition Date, Acquisition Time (UTC), Detection Confidence, Fire Radiative Power (FRP), Brightness Temperature.
- **Provider**: NASA Earthdata / FIRMS (Fire Information for Resource Management System).

### 1.2 Copernicus ERA5-Land Atmospheric Reanalysis
- **Producing Entity**: European Centre for Medium-Range Weather Forecasts (ECMWF).
- **Dataset**: ERA5-Land hourly gridded surface meteorological reanalysis.
- **Spatial Resolution**: 0.10° × 0.10° latitude-longitude regular grid (~9 km).
- **Variables Ingested**:
  - `2m_temperature` (Kelvin → °C): Ambient thermal forcing
  - `2m_dewpoint_temperature` (Kelvin): Humidity calculation
  - `relative_humidity` (%): Derived atmospheric moisture
  - `10m_wind_speed` (m/s): Wind ventilation and drying velocity
  - `surface_pressure` (Pa → hPa): Atmospheric barometric pressure
  - `total_precipitation` (m → mm): Rain accumulation
  - `volumetric_soil_water_layer_1` (m³/m³): Surface layer soil moisture (0–7 cm depth)
- **Aggregation Protocol**:
  - 1-day lag: Meteorological state over preceding 24 hours ($t - 24\text{h}$ to $t$).
  - 3-day antecedent window: Mean, min, max, total precipitation over preceding 72 hours.
  - 7-day antecedent window: Mean, min, max, total precipitation over preceding 168 hours.

### 1.3 Survey of India Sovereign Boundary
- **Source**: Official Survey of India administrative boundary.
- **Format**: GeoJSON polygon collection (`data/processed/india_boundary.geojson`).
- **Function**: Authoritative spatial masking ensuring all training, validation, testing, and live operational FIRMS detections lie strictly within sovereign Indian territory.

---

## 2. Dataset Construction Pipeline

```
[NASA FIRMS VIIRS 375m]                  [Copernicus ERA5-Land Reanalysis]
 2018–2025 Active Fires                   0.10° Hourly Surface Meteorology
          │                                              │
          ▼                                              ▼
[Spatial 0.1° Binning]                     [Multi-Timescale Aggregations]
Grid Centroid (Lat, Lon)                  1-Day Lag, 3-Day Window, 7-Day Window
          │                                              │
          ├───────────────────────┬──────────────────────┘
          │                       │
          ▼                       ▼
[Fire Cells (Y=1)]      [Controlled Non-Fire (Y=0)]
65,518 Detections       65,482 Matched Negative Samples
          │                       │
          └───────────┬───────────┘
                      ▼
       [Canonical 31-Feature Dataset]
     data/processed/india_fire_weather_final.csv
           (131,000 observations)
```

1. **Fire Cell Aggregation**: Continuous satellite fire detections are mapped onto a uniform 0.10° spatial grid and matched to their nearest acquisition hour.
2. **Negative Cell Sampling**: For every fire event, a non-detection grid cell is sampled from the active surveillance universe within matching temporal windows to build a controlled 1:1 case-control dataset.
3. **Meteorological Feature Extraction**: Hourly ERA5-Land series are extracted for each spatiotemporal grid cell, generating the 26 multi-timescale antecedent weather statistics.
4. **Coordinate Integration**: Spatial grid coordinates (`grid_lat`, `grid_lon`) and temporal indicators (`hour`, `year`, `month`) are appended to construct the final 31-feature vector.
