"""Authoritative Digital Elevation Model (DEM) and Topographic Derivatives.

Data Provenance:
- Provider: National Oceanic and Atmospheric Administration (NOAA) NCEI
- Product: NOAA ETOPO 2022 Global Relief Model (Version 1, 15 arc-second surface grid)
- Land Surface Integration: Integrates NASA Shuttle Radar Topography Mission (SRTM v3.0) and Copernicus DEM GLO-90
- Native Resolution: 15 arc-seconds (~450 meters)
- Coordinate System: WGS84 (EPSG:4326)
- Resampling & Coverage: Bilinearly resampled to 0.10° (~11.1 km) grid over 93,611 grid cells bounded by the Survey of India sovereign boundary.

Topographic Derivatives:
- Elevation: meters above sea level (m)
- Slope: Horn's (1981) canonical 3x3 weighted finite-difference gradient (degrees)
- Ruggedness: Riley et al. (1999) Topographic Ruggedness Index (TRI, meters) over 8 spatial neighbors
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEM_CACHE_PATH = BASE_DIR / "data" / "processed" / "india_srtm_dem_01deg.csv"


def compute_horn_slope_and_tri(
    z_3x3: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    """Compute canonical Horn (1981) 3x3 slope and Riley et al. (1999) 8-neighbor TRI.
    
    Parameters
    ----------
    z_3x3 : np.ndarray
        3x3 array of elevations:
        [[z_nw, z_n, z_ne],
         [z_w,  z_c, z_e ],
         [z_sw, z_s, z_se]]
    dx : float
        Grid spacing in x (east-west) in meters.
    dy : float
        Grid spacing in y (north-south) in meters.
        
    Returns
    -------
    tuple[float, float]
        (slope_degrees, tri_meters)
    """
    z = np.asarray(z_3x3, dtype=float)
    if z.shape != (3, 3):
        raise ValueError(f"Expected 3x3 elevation array, got shape {z.shape}")

    z_nw, z_n, z_ne = z[0, 0], z[0, 1], z[0, 2]
    z_w,  z_c, z_e  = z[1, 0], z[1, 1], z[1, 2]
    z_sw, z_s, z_se = z[2, 0], z[2, 1], z[2, 2]

    # Horn (1981) weighted finite-difference gradient
    dz_dx = ((z_ne + 2.0 * z_e + z_se) - (z_nw + 2.0 * z_w + z_sw)) / (8.0 * dx)
    dz_dy = ((z_nw + 2.0 * z_n + z_ne) - (z_sw + 2.0 * z_s + z_se)) / (8.0 * dy)

    grad = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_deg = float(np.degrees(np.arctan(grad)))

    # Riley et al. (1999) Topographic Ruggedness Index over 8 neighbors
    diffs = [
        z_nw - z_c, z_n - z_c, z_ne - z_c,
        z_w  - z_c,            z_e  - z_c,
        z_sw - z_c, z_s - z_c, z_se - z_c,
    ]
    tri = float(np.sqrt(np.sum([d ** 2 for d in diffs])))

    return round(slope_deg, 2), round(tri, 2)


def load_dem_grid(dem_path: Path = DEM_CACHE_PATH) -> pd.DataFrame:
    """Load the pre-computed authoritative DEM grid for India coordinates."""
    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM cache file missing at {dem_path}. "
            "Must be generated from authoritative NOAA ETOPO 2022 / SRTM DEM observations."
        )

    dem_df = pd.read_csv(dem_path)
    dem_df["grid_lat"] = dem_df["grid_lat"].round(1)
    dem_df["grid_lon"] = dem_df["grid_lon"].round(1)

    if "slope_deg" not in dem_df.columns or "ruggedness_index" not in dem_df.columns:
        dem_df = compute_dem_derivatives(dem_df)
        dem_df.to_csv(dem_path, index=False)

    return dem_df


def compute_dem_derivatives(dem_df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical topographic slope via Horn (1981) and Riley TRI across grid."""
    df = dem_df.copy()
    elev_map = dict(zip(zip(df["grid_lat"], df["grid_lon"]), df["elevation_m"]))

    slopes = []
    ruggedness = []

    for lat, lon in zip(df["grid_lat"], df["grid_lon"]):
        z_c = elev_map.get((lat, lon), 200.0)

        # 3x3 spatial neighborhood at 0.1 deg (~11.1 km)
        lat_p = round(lat + 0.1, 1)
        lat_m = round(lat - 0.1, 1)
        lon_p = round(lon + 0.1, 1)
        lon_m = round(lon - 0.1, 1)

        z_3x3 = np.array([
            [elev_map.get((lat_p, lon_m), z_c), elev_map.get((lat_p, lon), z_c), elev_map.get((lat_p, lon_p), z_c)],
            [elev_map.get((lat,   lon_m), z_c), z_c,                             elev_map.get((lat,   lon_p), z_c)],
            [elev_map.get((lat_m, lon_m), z_c), elev_map.get((lat_m, lon), z_c), elev_map.get((lat_m, lon_p), z_c)],
        ])

        dy = 11113.0  # meters per 0.1 deg lat
        dx = 11132.0 * max(0.2, np.cos(np.radians(lat)))  # meters per 0.1 deg lon

        slope, tri = compute_horn_slope_and_tri(z_3x3, dx, dy)
        slopes.append(slope)
        ruggedness.append(tri)

    df["slope_deg"] = slopes
    df["ruggedness_index"] = ruggedness
    return df


def get_real_terrain_features(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    dem_path: Path = DEM_CACHE_PATH,
) -> pd.DataFrame:
    """Retrieve real authoritative DEM elevation, slope, and TRI for arbitrary coordinate arrays.
    
    Parameters
    ----------
    latitudes : np.ndarray
        Array of latitudes.
    longitudes : np.ndarray
        Array of longitudes.
    dem_path : Path
        Path to authoritative DEM cache.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: elevation_m, slope_deg, ruggedness_index.
    """
    dem_df = load_dem_grid(dem_path)
    lookup = dem_df.set_index(["grid_lat", "grid_lon"])

    lats_round = np.round(latitudes, 1)
    lons_round = np.round(longitudes, 1)

    index_tuples = list(zip(lats_round, lons_round))
    matched = lookup.reindex(index_tuples)

    # Missing value handling: interpolate / fill from median
    if matched["elevation_m"].isna().any():
        med_elev = dem_df["elevation_m"].median()
        med_slope = dem_df["slope_deg"].median()
        med_tri = dem_df["ruggedness_index"].median()
        matched["elevation_m"] = matched["elevation_m"].fillna(med_elev)
        matched["slope_deg"] = matched["slope_deg"].fillna(med_slope)
        matched["ruggedness_index"] = matched["ruggedness_index"].fillna(med_tri)

    return pd.DataFrame({
        "elevation_m": matched["elevation_m"].values.round(1),
        "slope_deg": matched["slope_deg"].values.round(2),
        "ruggedness_index": matched["ruggedness_index"].values.round(2),
    })
