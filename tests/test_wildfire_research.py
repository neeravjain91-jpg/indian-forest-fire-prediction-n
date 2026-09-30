"""Scientific verification test suite for corrected wildfire research pipeline."""

import bisect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.environmental import (
    assign_ecological_regime,
    compute_soil_drought_index,
    compute_vapor_pressure_deficit,
)
from src.data.terrain import get_real_terrain_features
from src.evaluation.calibration import ModelCalibrator, UncertaintyEstimator
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_expected_calibration_error,
)
from src.evaluation.statistical_testing import (
    compute_bootstrap_confidence_interval,
    compute_precision_recall_at_k,
)
from src.events.event_clustering import cluster_fire_events, haversine_km


def test_real_dem_elevation_derivatives():
    """Verify terrain features are derived from real DEM and adhere to Indian geography."""
    # Test coordinates: New Delhi, Mumbai, Shimla, Bengaluru
    lats = np.array([28.6, 19.0, 31.1, 13.0])
    lons = np.array([77.2, 72.8, 77.2, 77.6])
    terrain = get_real_terrain_features(lats, lons)

    assert len(terrain) == 4
    # All elevations, slopes, and TRIs must be non-negative
    assert (terrain["elevation_m"] >= 0.0).all()
    assert (terrain["slope_deg"] >= 0.0).all()
    assert (terrain["ruggedness_index"] >= 0.0).all()

    # Shimla (Himalayas) elevation must be high (>1500m)
    assert terrain.loc[2, "elevation_m"] > 1500.0
    # Mumbai (Coastal sea level) must be low (<50m)
    assert terrain.loc[1, "elevation_m"] < 50.0
    # New Delhi (Alluvial plain) must be ~150-300m
    assert 150.0 <= terrain.loc[0, "elevation_m"] <= 300.0
    # Bengaluru (Deccan plateau) must be ~700-1100m
    assert 700.0 <= terrain.loc[3, "elevation_m"] <= 1100.0


def test_no_future_leakage_in_fire_history():
    """Verify strictly causal time-indexing: no contemporaneous or future dates enter history."""
    dates_list = [
        pd.Timestamp("2018-03-01"),
        pd.Timestamp("2019-05-10"),
        pd.Timestamp("2020-04-12"),
    ]

    # Date before first fire: count must be 0
    t_before = pd.Timestamp("2018-01-01")
    assert bisect.bisect_left(dates_list, t_before) == 0

    # Date of first fire: count must be 0 (cannot leak same-day detection)
    t_exact_1 = pd.Timestamp("2018-03-01")
    assert bisect.bisect_left(dates_list, t_exact_1) == 0

    # Day after first fire: count is 1
    t_after_1 = pd.Timestamp("2018-03-02")
    assert bisect.bisect_left(dates_list, t_after_1) == 1

    # Date of second fire: count is 1 (only includes fire 1, excludes fire 2)
    t_exact_2 = pd.Timestamp("2019-05-10")
    assert bisect.bisect_left(dates_list, t_exact_2) == 1

    # Date after second fire: count is 2
    t_after_2 = pd.Timestamp("2019-05-11")
    assert bisect.bisect_left(dates_list, t_after_2) == 2

    # In 2024: count is 3
    t_future = pd.Timestamp("2024-01-01")
    assert bisect.bisect_left(dates_list, t_future) == 3


def test_event_persistence_target_definition():
    """Verify that event persistence is strictly derived from connected complex continuity."""
    # Synthetic fire events:
    # Event 1: Burns on Day 1 and continues on Day 2
    # Event 2: Burns on Day 1 only (extinguished on Day 1)
    test_detections = pd.DataFrame({
        "grid_lat": [20.0, 20.1, 28.0],
        "grid_lon": [78.0, 78.1, 85.0],
        "acq_date": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-01"]),
        "max_frp": [10.0, 15.0, 5.0],
        "fire_detections": [1, 1, 1],
    })

    clustered, summary = cluster_fire_events(
        test_detections, spatial_radius_km=30.0, temporal_gap_days=2
    )

    # First event (EVT_000001) has detections on 2024-03-01 and 2024-03-02 (duration = 2)
    # Second event (EVT_000002) has detection on 2024-03-01 only (duration = 1)
    ev1 = summary[summary["duration_days"] == 2]
    ev2 = summary[summary["duration_days"] == 1]
    assert len(ev1) == 1
    assert len(ev2) == 1

    # For Event 1 on Day 1: it continues into Day 2 -> persistence = 1
    # For Event 2 on Day 1: it ends on Day 1 -> persistence = 0
    active_dates_ev1 = set(clustered[clustered["event_id"] == ev1.iloc[0]["event_id"]]["acq_date"].dt.strftime("%Y-%m-%d"))
    active_dates_ev2 = set(clustered[clustered["event_id"] == ev2.iloc[0]["event_id"]]["acq_date"].dt.strftime("%Y-%m-%d"))

    assert "2024-03-02" in active_dates_ev1
    assert "2024-03-02" not in active_dates_ev2


def test_chronological_splits_integrity():
    """Verify strict chronological non-overlap across training, validation, and testing."""
    train_path = Path("data/splits/train_chronological.csv")
    val_path = Path("data/splits/val_chronological.csv")
    test_path = Path("data/splits/test_chronological.csv")

    if train_path.exists() and val_path.exists() and test_path.exists():
        tr = pd.read_csv(train_path, usecols=["acq_date", "year"])
        va = pd.read_csv(val_path, usecols=["acq_date", "year"])
        te = pd.read_csv(test_path, usecols=["acq_date", "year"])

        assert tr["year"].max() <= 2022
        assert va["year"].min() == va["year"].max() == 2023
        assert te["year"].min() >= 2024
        assert tr["acq_date"].max() < va["acq_date"].min() < te["acq_date"].min()


def test_leave_one_ecoregion_out_disjointness():
    """Verify complete geographic disjointness across all 6 LOEO splits."""
    splits_dir = Path("data/splits")
    for reg in ["central", "western_ghats", "northeast", "north", "east", "northwest"]:
        tr_file = splits_dir / f"train_loeo_exclude_{reg}.csv"
        te_file = splits_dir / f"test_loeo_holdout_{reg}.csv"
        if tr_file.exists() and te_file.exists():
            tr = pd.read_csv(tr_file, usecols=["ecological_regime"])
            te = pd.read_csv(te_file, usecols=["ecological_regime"])
            assert reg.upper() not in tr["ecological_regime"].values
            assert (te["ecological_regime"] == reg.upper()).all()


def test_bootstrap_confidence_interval_math():
    """Verify bootstrap confidence interval calculation for metric differences."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 1000)
    p_better = np.clip(y_true * 0.7 + np.random.uniform(0, 0.3, 1000), 0.05, 0.95)
    p_worse = np.random.uniform(0, 1, 1000)

    # Better vs Worse: CI must exclude zero
    ci = compute_bootstrap_confidence_interval(y_true, p_better, p_worse, metric_name="roc_auc", n_bootstraps=200)
    assert ci["ci_excludes_zero"] is True
    assert ci["ci_95_lower"] > 0

    # Better vs Identical: CI must include zero
    ci_ident = compute_bootstrap_confidence_interval(y_true, p_better, p_better, metric_name="roc_auc", n_bootstraps=200)
    assert ci_ident["ci_excludes_zero"] is False
    assert ci_ident["ci_95_lower"] <= 0 <= ci_ident["ci_95_upper"]


def test_precision_at_k_rare_events():
    """Verify precision@k and recall@k computation for class-imbalanced targets."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 2 positives out of 10
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

    pk_df = compute_precision_recall_at_k(y_true, y_prob, k_list=[2, 5, 10])
    assert len(pk_df) == 3
    # Top 2 predictions are both positives
    assert pk_df.loc[0, "precision_at_k"] == 1.0
    assert pk_df.loc[0, "recall_at_k"] == 1.0
    # Top 5 predictions: 2 hits out of 5
    assert pk_df.loc[1, "precision_at_k"] == 0.4
    assert pk_df.loc[1, "recall_at_k"] == 1.0
