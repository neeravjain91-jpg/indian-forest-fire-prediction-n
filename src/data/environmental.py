"""Environmental covariates and fuel dryness dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vapor_pressure_deficit(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Compute atmospheric Vapor Pressure Deficit (VPD) in kPa using Tetens formula.
    
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
    """Compute topsoil moisture deficit / dryness index.
    
    Higher values indicate severely desiccated surface soil fuels.
    
    Parameters
    ----------
    soil_moisture : np.ndarray
        Volumetric soil moisture (m^3/m^3), typical range 0.05 to 0.45.
        
    Returns
    -------
    np.ndarray
        Normalized drought index in [0, 1].
    """
    sm = np.asarray(soil_moisture, dtype=np.float64)
    # Reference field capacity ~ 0.35 m^3/m^3
    field_capacity = 0.35
    deficit = np.clip((field_capacity - sm) / field_capacity, 0.0, 1.0)
    return np.round(deficit, 3)


def assign_ecological_regime(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Assign each coordinate to one of 6 cohesive Indian ecological fire regimes.
    
    Parameters
    ----------
    latitudes : np.ndarray
        Latitude array.
    longitudes : np.ndarray
        Longitude array.
        
    Returns
    -------
    np.ndarray of strings
        Ecological regime names:
        - 'NORTHEAST': Subtropical moist forests / Purvanchal
        - 'NORTH': Western Himalayas and Siwalik pine forests
        - 'WESTERN_GHATS': Western Ghats moist deciduous & evergreen
        - 'CENTRAL': Central Indian dry deciduous teak/sal belt
        - 'EAST': Eastern Ghats & Chota Nagpur plateau
        - 'NORTHWEST': Semi-arid Aravalli and thorn scrub
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
