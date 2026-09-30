"""Multimodal dataset construction with real DEM terrain, causal time-indexing, and event persistence.

Corrections implemented:
1. Real Copernicus/SRTM DEM terrain features (elevation, Horn slope, Riley TRI).
2. Strictly causal time-indexed fire history (no future leakage: all features t < T).
3. Connected-component event persistence target (1 only if active event continues into T+24h).
4. Leave-One-Ecoregion-Out (LOEO) and strict chronological split generation.
"""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.environmental import (
    assign_ecological_regime,
    compute_soil_drought_index,
    compute_vapor_pressure_deficit,
)
from src.data.terrain import get_real_terrain_features
from src.events.event_clustering import cluster_fire_events, run_threshold_sensitivity_study


def build_multimodal_dataset(
    input_path: Path,
    output_path: Path,
    events_output_path: Path,
    sensitivity_output_path: Path,
    splits_dir: Path,
) -> pd.DataFrame:
    """Build the corrected multimodal dataset with real DEM and causal history."""
    print(f"Loading base dataset from {input_path}...", flush=True)
    df = pd.read_csv(input_path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["grid_lat"] = df["grid_lat"].round(1)
    df["grid_lon"] = df["grid_lon"].round(1)
    df["year"] = df["acq_date"].dt.year.astype(int)
    df["month"] = df["acq_date"].dt.month.astype(int)

    n_initial = len(df)
    print(f"Base dataset loaded: {n_initial:,} records.", flush=True)

    # 1. Real Copernicus / SRTM DEM Terrain Features
    print("Assigning real authoritative Copernicus/SRTM DEM elevation, slope, and TRI...", flush=True)
    terrain_df = get_real_terrain_features(df["grid_lat"].values, df["grid_lon"].values)
    for col in ["elevation_m", "slope_deg", "ruggedness_index"]:
        df[col] = terrain_df[col].values
    print(
        f"Real DEM loaded: Mean Elevation = {df['elevation_m'].mean():.1f}m, "
        f"Mean Slope = {df['slope_deg'].mean():.2f}°, Mean TRI = {df['ruggedness_index'].mean():.2f}",
        flush=True,
    )

    # 2. Atmospheric & Fuel Dryness Dynamics
    print("Computing environmental and fuel dryness dynamics...", flush=True)
    df["vpd_1d"] = compute_vapor_pressure_deficit(df["temp_1d"].values, df["rh_1d"].values)
    df["vpd_3d_mean"] = compute_vapor_pressure_deficit(df["temp_3d_mean"].values, df["rh_3d_mean"].values)
    df["soil_drought_index"] = compute_soil_drought_index(df["soil_1d"].values)
    df["ecological_regime"] = assign_ecological_regime(df["grid_lat"].values, df["grid_lon"].values)

    # 3. Spatiotemporal Fire Event Clustering
    print("Clustering active fire detections into coherent fire events...", flush=True)
    fire_subset = df[df["fire"] == 1][["grid_lat", "grid_lon", "acq_date", "max_frp", "fire_detections"]].copy()

    sensitivity_df = run_threshold_sensitivity_study(fire_subset)
    sensitivity_output_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_df.to_csv(sensitivity_output_path, index=False)
    print(f"Sensitivity study saved to {sensitivity_output_path}.", flush=True)

    clustered_fires, events_summary = cluster_fire_events(
        fire_subset, spatial_radius_km=25.0, temporal_gap_days=2
    )
    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    events_summary.to_csv(events_output_path, index=False)
    print(f"Constructed {len(events_summary):,} unique spatiotemporal fire events.", flush=True)

    # Build event mapping for detections
    fire_event_map = dict(
        zip(
            zip(
                clustered_fires["grid_lat"].round(1),
                clustered_fires["grid_lon"].round(1),
                clustered_fires["acq_date"].dt.strftime("%Y-%m-%d"),
            ),
            clustered_fires["event_id"],
        )
    )

    # Build index of event active dates
    event_active_dates = {}
    for eid, grp in clustered_fires.groupby("event_id"):
        event_active_dates[eid] = set(grp["acq_date"].dt.strftime("%Y-%m-%d"))

    # 4. Strictly Causal Time-Indexed Fire History Features (Observation Time < T)
    print("Computing strictly causal time-indexed fire recurrence (guaranteed t < T)...", flush=True)
    # Group all fire detections by cell with sorted date lists
    cell_fire_dates: dict[tuple[float, float], list[pd.Timestamp]] = {}
    for _, r in fire_subset.iterrows():
        key = (round(float(r["grid_lat"]), 1), round(float(r["grid_lon"]), 1))
        cell_fire_dates.setdefault(key, []).append(r["acq_date"])

    for key in cell_fire_dates:
        cell_fire_dates[key].sort()

    base_date = pd.Timestamp("2018-01-01")
    recurrence_values = []
    antecedent_24h = []

    for lat, lon, date in zip(df["grid_lat"], df["grid_lon"], df["acq_date"]):
        key = (round(float(lat), 1), round(float(lon), 1))
        dates_list = cell_fire_dates.get(key, [])

        # bisect_left returns number of elements strictly less than date
        prior_fire_count = bisect.bisect_left(dates_list, date)

        # Elapsed observation time in years (normalized)
        days_elapsed = max(30, (date - base_date).days)
        years_elapsed = days_elapsed / 365.25
        rate = prior_fire_count / years_elapsed
        recurrence_values.append(round(float(rate), 4))

        # Antecedent fire 24h: did this cell burn on date - 1 day?
        prev_day = date - pd.Timedelta(days=1)
        idx_prev = bisect.bisect_left(dates_list, prev_day)
        has_fire_yesterday = 1 if (idx_prev < len(dates_list) and dates_list[idx_prev] == prev_day) else 0
        antecedent_24h.append(has_fire_yesterday)

    df["fire_history_recurrence"] = recurrence_values
    df["antecedent_fire_24h"] = antecedent_24h

    # 5. Connected-Component Event Persistence Target & Multi-Horizon Targets
    print("Constructing mathematically defined forward targets and event persistence...", flush=True)
    tomorrow_dates = (df["acq_date"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    two_days_dates = (df["acq_date"] + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")

    all_fire_cell_dates = set(
        zip(
            fire_subset["grid_lat"].round(1),
            fire_subset["grid_lon"].round(1),
            fire_subset["acq_date"].dt.strftime("%Y-%m-%d"),
        )
    )

    df["target_fire_lead_24h"] = [
        1 if (lat, lon, td) in all_fire_cell_dates else 0
        for lat, lon, td in zip(df["grid_lat"], df["grid_lon"], tomorrow_dates)
    ]
    df["target_fire_lead_48h"] = [
        1 if (lat, lon, td) in all_fire_cell_dates else 0
        for lat, lon, td in zip(df["grid_lat"], df["grid_lon"], two_days_dates)
    ]

    # Event persistence: For active fires at T, does THAT connected event continue into T+1d?
    persistence = []
    for lat, lon, date, fire_label, tom in zip(
        df["grid_lat"], df["grid_lon"], df["acq_date"], df["fire"], tomorrow_dates
    ):
        if fire_label == 1:
            det_key = (lat, lon, date.strftime("%Y-%m-%d"))
            eid = fire_event_map.get(det_key)
            if eid and tom in event_active_dates.get(eid, set()):
                persistence.append(1)
            else:
                persistence.append(0)
        else:
            persistence.append(0)

    df["target_event_persistence"] = persistence

    # Save corrected dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Corrected multimodal dataset saved to {output_path} ({len(df):,} records).", flush=True)

    # 6. Generate Clean Splits
    splits_dir.mkdir(parents=True, exist_ok=True)
    print("Generating corrected chronological and Leave-One-Ecoregion-Out (LOEO) splits...", flush=True)

    # Chronological
    train_chrono = df[df["year"] <= 2022].copy()
    val_chrono = df[df["year"] == 2023].copy()
    test_chrono = df[df["year"] >= 2024].copy()

    train_chrono.to_csv(splits_dir / "train_chronological.csv", index=False)
    val_chrono.to_csv(splits_dir / "val_chronological.csv", index=False)
    test_chrono.to_csv(splits_dir / "test_chronological.csv", index=False)

    # Generate LOEO splits for all 6 regions
    for reg in ["CENTRAL", "WESTERN_GHATS", "NORTHEAST", "NORTH", "EAST", "NORTHWEST"]:
        train_reg = df[df["ecological_regime"] != reg].copy()
        test_reg = df[df["ecological_regime"] == reg].copy()
        train_reg.to_csv(splits_dir / f"train_loeo_exclude_{reg.lower()}.csv", index=False)
        test_reg.to_csv(splits_dir / f"test_loeo_holdout_{reg.lower()}.csv", index=False)

    print(
        f"Chronological split: Train={len(train_chrono):,}, Val={len(val_chrono):,}, Test={len(test_chrono):,}. "
        "Generated all 6 LOEO geographic splits.",
        flush=True,
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/india_fire_weather_final.csv")
    p.add_argument("--output", default="data/features/multimodal_features.csv")
    p.add_argument("--events-output", default="data/events/fire_events.csv")
    p.add_argument("--sensitivity-output", default="data/events/clustering_sensitivity.csv")
    p.add_argument("--splits-dir", default="data/splits")
    args = p.parse_args()

    build_multimodal_dataset(
        Path(args.input),
        Path(args.output),
        Path(args.events_output),
        Path(args.sensitivity_output),
        Path(args.splits_dir),
    )


if __name__ == "__main__":
    main()
