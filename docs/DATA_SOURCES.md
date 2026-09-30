# Authoritative Data Sources & Preprocessing Protocols

## 1. Primary Data Sources

| Modality / Source | Product / Provider | Spatial Resolution | Temporal Cadence | Geographic Coverage | License / Terms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Active Fire Telemetry** | VIIRS 375m NRT (`VNP14IMGTDL_NRT`, `VJ114IMGTDL_NRT`, `VJ214IMGTDL_NRT`) via NASA FIRMS | 375 m at nadir | Polar orbit overpasses (~01:30 & 13:30 local solar time) | Sovereign India ($6^\circ\text{N} - 38^\circ\text{N}$, $68^\circ\text{E} - 98^\circ\text{E}$) | NASA Open Data Policy (Free, Attribution required) |
| **Meteorological Reanalysis** | ERA5-Land (ECMWF / Copernicus Climate Change Service) | 0.10° (~11.1 km) | Hourly, aggregated to 24h, 72h, 168h windows | Pan-India Land surface | Copernicus Open Access License |
| **Terrain Geomorphology** | NOAA ETOPO 2022 Global Relief Model v1 (integrating NASA SRTM v3.0 land elevation) via NOAA NCEI | 15 arc-seconds (~450m native, resampled to 0.10°) | Static (Topographic baseline) | Sovereign Indian territory | Public Domain / NOAA NCEI Open Data |
| **Fuel / Dryness Proxies** | Derived from ERA5-Land atmospheric state & soil moisture | 0.10° (~11.1 km) | Multi-day antecedent windows | Pan-India Land surface | Derived secondary product |
| **Administrative / Boundary** | Survey of India / Datameet Open Boundary Project | Vector polygons | Official national boundary | Sovereign territory of India | Open Data Commons (ODC-BY) |

---

## 2. Ingestion & Preprocessing Protocols

### A. Satellite Active Fire Telemetry (VIIRS 375m)
* **Aggregation to Analysis Grid**: Raw VIIRS 375m detections contain latitude, longitude, acquisition date, acquisition time (UTC), brightness temperature, confidence category (`l`, `n`, `h`), and Fire Radiative Power (FRP in MW).
* **Spatial Binning**: Active detections are binned into regular $0.10^\circ \times 0.10^\circ$ grid cells ($\approx 11.1 \times 11.1\text{ km}$ at the equator).
* **Information Extraction**:
  - `fire_detections`: Total count of VIIRS fire pixels within the 0.10° cell on that acquisition cycle.
  - `mean_confidence`: Normalized detection confidence score ($l=0.25, n=0.50, h=1.00$).
  - `max_frp`: Maximum fire radiative power observed within the cell during the observation cycle.
* **Leakage Gating**: FRP, detection count, and brightness temperature recorded at time $T$ are **strictly prohibited** from serving as input predictors for the occurrence label at time $T$ or future times. They are reserved for ground-truth target construction (e.g., event intensity) and descriptive characterization.

### B. ERA5-Land Atmospheric Dynamics
* **Core Variables**:
  - `temperature_2m` ($^\circ\text{C}$): Air temperature at 2 meters.
  - `relative_humidity_2m` ($\%$): Computed from 2m temperature and dewpoint temperature using Magnus-Tetens vapor pressure formulation.
  - `surface_pressure` ($\text{hPa}$): Atmospheric surface pressure.
  - `wind_speed_10m` ($\text{m/s}$): Horizontal wind speed vector magnitude at 10 meters.
  - `soil_moisture_0_to_7cm` ($\text{m}^3/\text{m}^3$): Volumetric soil water content in the topsoil layer.
  - `precipitation` ($\text{mm}$): Total liquid and solid water reaching the surface.
* **Temporal Windows**:
  - **1-Day ($24\text{h}$)**: Mean temperature, relative humidity, wind speed, surface pressure, soil moisture, and total precipitation over the antecedent 24 hours ($T-24\text{h} \to T$).
  - **3-Day ($72\text{h}$)**: Mean, max, min temperature; mean, min relative humidity; mean, max wind speed; mean pressure; mean soil moisture; total precipitation over 72 hours.
  - **7-Day ($168\text{h}$)**: Cumulative multi-day antecedent drying indicators (mean, max, min temp; mean, min RH; mean, max wind; total rain; mean soil moisture).

### C. Digital Elevation Model (NOAA ETOPO 2022)
* **Data Provenance**:
  - Provider: National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Information (NCEI).
  - Product: NOAA ETOPO 2022 Global Relief Model (Version 1, 15 arc-second surface elevation grid).
  - Terrestrial Topography Source: Integrates NASA Shuttle Radar Topography Mission (SRTM v3.0) and Copernicus DEM GLO-90 over land surfaces.
  - Aggregation: Bilinearly resampled to 0.10° (~11.1 km) grid centroids across 93,611 grid cells bounded by the official Survey of India boundary.
* **Topographic Derivatives**:
  - `elevation_m`: Mean surface elevation above sea level (meters).
  - `slope_deg`: Topographic slope computed via canonical 3x3 weighted finite-difference gradient (Horn, 1981).
  - `ruggedness_index`: Topographic Ruggedness Index (TRI) computed via Riley et al. (1999) root-sum-square elevation variance across all 8 spatial neighbors.

### D. Derived Fuel Moisture & Vapor Pressure Deficit (VPD)
* **VPD Estimation**:
  $$e_s(T) = 0.61078 \exp\left(\frac{17.27 \cdot T}{T + 237.3}\right)\quad (\text{kPa})$$
  $$\text{VPD} = e_s(T) \cdot \left(1 - \frac{\text{RH}}{100}\right)\quad (\text{kPa})$$
  VPD represents atmospheric evaporative demand and is a recognized physical driver of fine fuel moisture desiccation. Multi-day `vpd_3d_mean` is computed as an aggregate proxy from 3-day mean temperature and relative humidity ($VPD(\bar{T}, \overline{RH})$), noting the mild underestimation of diurnal peak VPD implied by Jensen's inequality as a consistent multi-timescale index.
