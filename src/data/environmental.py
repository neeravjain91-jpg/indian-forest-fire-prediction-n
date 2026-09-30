"""Environmental covariates, atmospheric drying dynamics, and predefined geographic regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vapor_pressure_deficit(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Compute atmospheric Vapor Pressure Deficit (VPD) in kPa using Tetens formulation.
    
    Formula:
      e_s(T) = 0.61078 * exp((17.27 * T) / (T + 237.3))
      e_a = e_s * (RH / 100)
      VPD = max(0, e_s - e_a)

    Scientific Note on Multi-Day Aggregation:
      When evaluated on multi-day mean temperature T_bar and relative humidity RH_bar,
      this yields an aggregate drying proxy VPD(T_bar, RH_bar). Because e_s(T) is strictly
      convex in T, Jensen's inequality implies E[VPD(T, RH)] >= VPD(E[T], E[RH]). This
      mildly dampens diurnal peak extremes while providing a consistent, smooth multi-timescale
      atmospheric moisture deficit index.
    
    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature in degrees Celsius.
    rh_pct : np.ndarray
        Relative humidity in percentage (0 to 100).
        
    Returns
    -------
    np.ndarray
        Vapor pressure deficit in kilopascals (kPa).
    """
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.clip(np.asarray(rh_pct, dtype=np.float64), 0.0, 100.0)
    # Saturated vapor pressure (kPa)
    es = 0.61078 * np.exp((17.27 * t) / (t + 237.3))
    # Actual vapor pressure
    ea = es * (rh / 100.0)
    vpd = np.maximum(0.0, es - ea)
    return np.round(vpd, 3)


def compute_soil_drought_index(soil_moisture: np.ndarray) -> np.ndarray:
    """Compute surface soil moisture deficit proxy relative to nominal reference threshold.
    
    Formula:
      deficit = clip((theta_ref - theta) / theta_ref, 0.0, 1.0)
      where theta_ref = 0.35 m^3/m^3.
      
    Scientific Note:
      A uniform reference threshold (0.35 m^3/m^3) is used across India's modelling grid.
      This feature functions as a relative topsoil desiccation proxy rather than a localized
      soil-survey hydraulic field capacity (which varies geographically between sandy and
      clay-rich vertisols).
      
    Parameters
    ----------
    soil_moisture : np.ndarray
        Volumetric surface soil moisture (m^3/m^3).
        
    Returns
    -------
    np.ndarray
        Normalized soil moisture deficit proxy in [0, 1].
    """
    sm = np.asarray(soil_moisture, dtype=np.float64)
    field_capacity = 0.35
    deficit = np.clip((field_capacity - sm) / field_capacity, 0.0, 1.0)
    return np.round(deficit, 3)


def assign_geographic_regime(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Assign each coordinate to one of 6 predefined geographic fire regimes.
    
    Methodological Provenance:
      These six regimes are predefined latitudinal-longitudinal macro-climatic partitions
      designed to evaluate out-of-distribution spatial generalization. They represent
      broad geographic regimes rather than official WWF Terrestrial Ecoregion or WII
      biogeographic polygon boundaries.
    
    Regimes:
      - 'NORTHEAST': East of 88°E, lat >= 21°N (Purvanchal & Brahmaputra basin)
      - 'NORTH': lat >= 28.0°N, lon < 88.0°E (Himalayan montane & foothills)
      - 'WESTERN_GHATS': lat <= 21.0°N, lon <= 77.0°E (Moist western coastal escarpment)
      - 'CENTRAL': Interior Deccan plateau (default core)
      - 'EAST': 16°N <= lat < 28°N, 82.5°E <= lon < 88.0°E (Eastern Ghats / Chota Nagpur)
      - 'NORTHWEST': 21°N <= lat < 28°N, lon < 77.0°E (Semi-arid Thar / Aravalli scrub)
    """
    lats = np.asarray(latitudes, dtype=np.float64)
    lons = np.asarray(longitudes, dtype=np.float64)
    n = len(lats)
    regimes = np.full(n, "CENTRAL", dtype=object)

    # 1. Northeast (East of 88°E, lat >= 21°N)
    ne_mask = (lons >= 88.0) & (lats >= 21.0)
    regimes[ne_mask] = "NORTHEAST"

    # 2. Northern Himalayan Montane Belt (lat >= 28.0°N, lon < 88.0°E)
    north_mask = (lats >= 28.0) & (lons < 88.0)
    regimes[north_mask] = "NORTH"

    # 3. Western Ghats (lat <= 21.0°N, lon <= 77.0°E)
    wg_mask = (lats <= 21.0) & (lons <= 77.0)
    regimes[wg_mask] = "WESTERN_GHATS"

    # 4. Eastern Zone (16°N <= lat < 28°N, 82.5°E <= lon < 88.0°E)
    east_mask = (lats >= 16.0) & (lats < 28.0) & (lons >= 82.5) & (lons < 88.0)
    regimes[east_mask] = "EAST"

    # 5. Northwest Semi-Arid (21°N <= lat < 28°N, lon < 77.0°E)
    nw_mask = (lats >= 21.0) & (lats < 28.0) & (lons < 77.0)
    regimes[nw_mask] = "NORTHWEST"

    return regimes


# Backward compatibility alias
assign_ecological_regime = assign_geographic_regime

