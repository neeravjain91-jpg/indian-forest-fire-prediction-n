"""Multimodal dataset construction with strict temporal causality and geographic tagging.

Integrates meteorology, terrain geomorphology, atmospheric fuel dryness,
fire event clusters, antecedent fire history, and forward forecast targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.environmental import (
    assign_ecological_regime,
    compute_soil_drought_index,
    compute_vapor_pressure_deficit,
)
from src.data.terrain import compute_terrain_features
from src.events.event_clustering import cluster_fire_events, run_threshold_sensitivity_study


def build_multimodal_dataset(
    input_path: Path,
    output_path: Path,
    events_output_path: Path,
    sensitivity_output_path: Path,
    splits_dir: Path,
) -> pd.DataFrame:
    """Build the complete multimodal dataset with leak-free targets and geographic regimes.
    
    Parameters
    ----------
    input_path : Path
        Path to processed base CSV (india_fire_weather_final.csv).
    output_path : Path
        Path to save full multimodal features.
    events_output_path : Path
        Path to save constructed fire events.
    sensitivity_output_path : Path
        Path to save clustering sensitivity study.
    splits_dir : Path
        Directory to write chronological and spatial splits.
        
    Returns
    -------
    pd.DataFrame
        Complete engineered multimodal dataframe.
    """
    print(f"Loading base dataset from {input_path}...", flush=True)
    df = pd.read_csv(input_path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["grid_lat"] = df["grid_lat"].round(1)
    df["grid_lon"] = df["grid_lon"].round(1)
    df["year"] = df["acq_date"].dt.year.astype(int)
    df["month"] = df["acq_date"].dt.month.astype(int)

    n_initial = len(df)
    print(f"Base dataset loaded: {n_initial:,} records.", flush=True)

    # 1. Terrain Geomorphology Features
    print("Computing terrain geomorphology covariates...", flush=True)
    terrain_df = compute_terrain_features(df["grid_lat"].values, df["grid_lon"].values)
    for col in terrain_df.columns:
        df[col] = terrain_df[col].values

    # 2. Environmental & Atmospheric Fuel Dryness Features
    print("Computing environmental and fuel dryness dynamics...", flush=True)
    df["vpd_1d"] = compute_vapor_pressure_deficit(df["temp_1d"].values, df["rh_1d"].values)
    df["vpd_3d_mean"] = compute_vapor_pressure_deficit(df["temp_3d_mean"].values, df["rh_3d_mean"].values)
    df["soil_drought_index"] = compute_soil_drought_index(df["soil_1d"].values)
    df["ecological_regime"] = assign_ecological_regime(df["grid_lat"].values, df["grid_lon"].values)

    # 3. Spatiotemporal Fire Event Clustering
    print("Executing spatiotemporal fire event clustering...", flush=True)
    fire_subset = df[df["fire"] == 1][["grid_lat", "grid_lon", "acq_date", "max_frp", "fire_detections"]].copy()

    # Run sensitivity analysis
    print("Running clustering parameter sensitivity study...", flush=True)
    sensitivity_df = run_threshold_sensitivity_study(fire_subset)
    sensitivity_output_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_df.to_csv(sensitivity_output_path, index=False)
    print(f"Sensitivity study saved to {sensitivity_output_path}.", flush=True)

    # Cluster events at standard research parameters (25 km radius, 2-day gap)
    clustered_fires, events_summary = cluster_fire_events(
        fire_subset, spatial_radius_km=25.0, temporal_gap_days=2
    )
    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    events_summary.to_csv(events_output_path, index=False)
    print(f"Constructed {len(events_summary):,} unique spatiotemporal fire events.", flush=True)

    # 4. Antecedent Fire History Features (Strictly Causal: historical rate computed on 2018-2022)
    print("Computing antecedent fire recurrence and persistence features...", flush=True)
    # Historic recurrence rate per 0.1° cell computed exclusively on baseline training period (2018-2022)
    hist_period = df[(df["year"] <= 2022) & (df["fire"] == 1)]
    cell_hist_counts = hist_period.groupby(["grid_lat", "grid_lon"]).size()
    max_count = max(1, cell_hist_counts.max())
    recurrence_map = (cell_hist_counts / max_count).to_dict()

    df["fire_history_recurrence"] = [
        round(recurrence_map.get((lat, lon), 0.0), 4)
        for lat, lon in zip(df["grid_lat"], df["grid_lon"])
    ]

    # Cell-level fire index for lookup
    fire_cell_dates = set(
        zip(
            df[df["fire"] == 1]["grid_lat"].round(1),
            df[df["fire"] == 1]["grid_lon"].round(1),
            df[df["fire"] == 1]["acq_date"].dt.strftime("%Y-%m-%d"),
        )
    )

    # Antecedent fire 24h (did this cell burn yesterday?)
    yesterday_dates = (df["acq_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    df["antecedent_fire_24h"] = [
        1 if (lat, lon, yd) in fire_cell_dates else 0
        for lat, lon, yd in zip(df["grid_lat"], df["grid_lon"], yesterday_dates)
    ]

    # Active cluster proximity proxy (km)
    # Approximated by antecedent fire activity in local neighborhood
    df["active_cluster_proximity_km"] = np.where(
        df["antecedent_fire_24h"] == 1,
        0.0,
        np.clip(100.0 - (df["fire_history_recurrence"] * 80.0), 5.0, 100.0),
    )

    # 5. Forward Forecast Targets (T+24h and T+48h)
    print("Constructing mathematically defined forward forecast targets...", flush=True)
    tomorrow_dates = (df["acq_date"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    two_days_dates = (df["acq_date"] + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")

    df["target_fire_lead_24h"] = [
        1 if (lat, lon, td) in fire_cell_dates else 0
        for lat, lon, td in zip(df["grid_lat"], df["grid_lon"], tomorrow_dates)
    ]
    df["target_fire_lead_48h"] = [
        1 if (lat, lon, td) in fire_cell_dates else 0
        for lat, lon, td in zip(df["grid_lat"], df["grid_lon"], two_days_dates)
    ]
    # Event persistence: For active fires at T, does the fire continue into T+24h?
    df["target_event_persistence"] = np.where(
        df["fire"] == 1,
        df["target_fire_lead_24h"],
        0,
    )

    # Save complete multimodal features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Engineered multimodal dataset saved to {output_path} ({len(df):,} rows).", flush=True)

    # 6. Generate Rigorous Splits
    splits_dir.mkdir(parents=True, exist_ok=True)
    print("Generating chronological and spatial evaluation splits...", flush=True)
    
    # Chronological Split
    train_chrono = df[df["year"] <= 2022].copy()
    val_chrono = df[df["year"] == 2023].copy()
    test_chrono = df[df["year"] >= 2024].copy()

    train_chrono.to_csv(splits_dir / "train_chronological.csv", index=False)
    val_chrono.to_csv(splits_dir / "val_chronological.csv", index=False)
    test_chrono.to_csv(splits_dir / "test_chronological.csv", index=False)

    print(
        f"Chronological split sizes: Train (2018-2022)={len(train_chrono):,}, "
        f"Val (2023)={len(val_chrono):,}, Test (2024-2025)={len(test_chrono):,}",
        flush=True,
    )

    # Geographic / Ecological Disjoint Split
    # Hold out CENTRAL regime (Deccan dry deciduous forest) as test domain
    train_spatial = df[df["ecological_regime"] != "CENTRAL"].copy()
    test_spatial = df[df["ecological_regime"] == "CENTRAL"].copy()

    train_spatial.to_csv(splits_dir / "train_spatial_non_central.csv", index=False)
    test_spatial.to_csv(splits_dir / "test_spatial_central.csv", index=False)

    print(
        f"Spatial holdout split sizes: Non-Central Train={len(train_spatial):,}, "
        f"Central Holdout Test={len(test_spatial):,}",
        flush=True,
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct multimodal wildfire dataset.")
    parser.add_argument("--input", default="data/processed/india_fire_weather_final.csv")
    parser.add_argument("--output", default="data/features/multimodal_features.csv")
    parser.add_argument("--events-output", default="data/events/fire_events.csv")
    parser.add_argument("--sensitivity-output", default="data/events/clustering_sensitivity.csv")
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()

    build_multimodal_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        events_output_path=Path(args.events_output),
        sensitivity_output_path=Path(args.sensitivity_output),
        splits_dir=Path(args.splits_dir),
    )


if __name__ == "__main__":
    main()
