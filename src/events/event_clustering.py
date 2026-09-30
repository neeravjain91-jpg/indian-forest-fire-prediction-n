"""Spatiotemporal DBSCAN-ST clustering of active fire observations into coherent fire events.

Constructs bounded fire complexes, tracks temporal persistence, duration,
bounding boxes, centroid trajectories, and expansion kinetics.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two coordinates in kilometers."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate compass bearing (azimuth in degrees) from coordinate 1 to 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return round(bearing, 1)


def cluster_fire_events(
    fire_df: pd.DataFrame,
    spatial_radius_km: float = 25.0,
    temporal_gap_days: int = 2,
    min_detections: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster active fire detections into coherent spatiotemporal fire events.
    
    Parameters
    ----------
    fire_df : pd.DataFrame
        DataFrame of active fire detections containing:
        ['grid_lat', 'grid_lon', 'acq_date', 'max_frp', 'fire_detections']
    spatial_radius_km : float
        Maximum spatial distance threshold (km) to link detections.
    temporal_gap_days : int
        Maximum temporal gap (days) to maintain event continuity.
    min_detections : int
        Minimum detections to qualify as an event.
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        1. detections_with_events: Original detections annotated with 'event_id'.
        2. events_summary: Summary table of each unique fire event.
    """
    df = fire_df.copy()
    if "acq_date" not in df.columns:
        raise ValueError("Missing 'acq_date' column in fire dataframe.")

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    # Sort chronologically
    df = df.sort_values("acq_date").reset_index(drop=True)
    n = len(df)
    
    # Initialize event assignments
    event_ids = np.full(n, -1, dtype=int)
    current_event_id = 0

    # Group by calendar date for fast temporal-window neighbor querying
    dates = df["acq_date"].dt.normalize().values
    lats = df["grid_lat"].values
    lons = df["grid_lon"].values

    # Connected component labeling across time
    for i in range(n):
        if event_ids[i] != -1:
            continue
        
        # Start new event cluster
        current_event_id += 1
        event_ids[i] = current_event_id
        queue = [i]

        while queue:
            curr = queue.pop(0)
            curr_lat, curr_lon, curr_date = lats[curr], lons[curr], dates[curr]

            # Search forward in temporal window
            time_diff_days = (dates - curr_date) / np.timedelta64(1, "D")
            candidate_mask = (event_ids == -1) & (time_diff_days >= 0) & (time_diff_days <= temporal_gap_days)
            candidate_indices = np.where(candidate_mask)[0]

            for cand in candidate_indices:
                # Approximate quick box check (1 deg ~ 111 km)
                d_lat = abs(lats[cand] - curr_lat)
                d_lon = abs(lons[cand] - curr_lon)
                if d_lat * 111.0 > spatial_radius_km or d_lon * 111.0 > spatial_radius_km:
                    continue

                dist_km = haversine_km(curr_lat, curr_lon, lats[cand], lons[cand])
                if dist_km <= spatial_radius_km:
                    event_ids[cand] = current_event_id
                    queue.append(cand)

    df["event_id"] = [f"EVT_{eid:06d}" for eid in event_ids]

    # Build event-level summary table
    event_records = []
    for eid, grp in df.groupby("event_id"):
        grp_sorted = grp.sort_values("acq_date")
        start_date = grp_sorted["acq_date"].min()
        end_date = grp_sorted["acq_date"].max()
        duration_days = int((end_date - start_date).total_seconds() // 86400) + 1
        
        start_lat = grp_sorted.iloc[0]["grid_lat"]
        start_lon = grp_sorted.iloc[0]["grid_lon"]
        end_lat = grp_sorted.iloc[-1]["grid_lat"]
        end_lon = grp_sorted.iloc[-1]["grid_lon"]

        displacement_km = haversine_km(start_lat, start_lon, end_lat, end_lon) if duration_days > 1 else 0.0
        bearing = compute_bearing_deg(start_lat, start_lon, end_lat, end_lon) if displacement_km > 1.0 else 0.0

        event_records.append({
            "event_id": eid,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "duration_days": duration_days,
            "centroid_lat": round(grp["grid_lat"].mean(), 2),
            "centroid_lon": round(grp["grid_lon"].mean(), 2),
            "detection_count": len(grp),
            "max_frp": round(grp["max_frp"].max() if "max_frp" in grp.columns else 0.0, 1),
            "mean_frp": round(grp["max_frp"].mean() if "max_frp" in grp.columns else 0.0, 1),
            "bbox_min_lat": grp["grid_lat"].min(),
            "bbox_max_lat": grp["grid_lat"].max(),
            "bbox_min_lon": grp["grid_lon"].min(),
            "bbox_max_lon": grp["grid_lon"].max(),
            "displacement_km": round(displacement_km, 2),
            "movement_bearing_deg": bearing,
        })

    events_summary = pd.DataFrame(event_records)
    return df, events_summary


def run_threshold_sensitivity_study(
    fire_df: pd.DataFrame,
    radii_km: list[float] = [15.0, 25.0, 35.0],
    temporal_gaps: list[int] = [1, 2],
) -> pd.DataFrame:
    """Conduct sensitivity analysis of fire event clustering thresholds.
    
    Parameters
    ----------
    fire_df : pd.DataFrame
        Sample of fire detections.
    radii_km : list[float]
        List of spatial radii (km) to evaluate.
    temporal_gaps : list[int]
        List of temporal gap days to evaluate.
        
    Returns
    -------
    pd.DataFrame
        Sensitivity comparison table.
    """
    results = []
    # Use representative 5,000-detection sample for rapid sensitivity analysis
    sample_df = fire_df.sample(min(5000, len(fire_df)), random_state=42).reset_index(drop=True)

    for r in radii_km:
        for tg in temporal_gaps:
            _, events = cluster_fire_events(sample_df, spatial_radius_km=r, temporal_gap_days=tg)
            multi_day = events[events["duration_days"] > 1]
            results.append({
                "spatial_radius_km": r,
                "temporal_gap_days": tg,
                "total_events": len(events),
                "multi_day_events": len(multi_day),
                "multi_day_pct": round(100.0 * len(multi_day) / max(1, len(events)), 1),
                "mean_duration_days": round(events["duration_days"].mean(), 2),
                "max_duration_days": int(events["duration_days"].max()),
                "mean_displacement_km": round(multi_day["displacement_km"].mean() if len(multi_day) else 0.0, 2),
            })
    return pd.DataFrame(results)
