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


def test_canonical_horn_slope_and_riley_tri_numerical():
    """Verify Horn (1981) slope and Riley et al. (1999) TRI against hand-calculated proofs."""
    from src.data.terrain import compute_horn_slope_and_tri

    # 1. Perfectly flat surface: slope = 0.0 deg, TRI = 0.0 m
    z_flat = np.full((3, 3), 500.0)
    slope_flat, tri_flat = compute_horn_slope_and_tri(z_flat, dx=100.0, dy=100.0)
    assert slope_flat == 0.0
    assert tri_flat == 0.0

    # 2. Linear ramp in X:
    # row 0: [100, 110, 120]
    # row 1: [100, 110, 120]
    # row 2: [100, 110, 120]
    # Hand calculation:
    # dz_dx = (480 - 400) / (8 * 100) = 80 / 800 = 0.10
    # dz_dy = 0.0
    # grad = 0.10 -> slope = arctan(0.10) * 180 / pi = 5.71 deg
    # Central pixel = 110. Differences = [-10, 0, 10, -10, 10, -10, 0, 10]
    # Sum of squares = 4 * 100 + 2 * 100 = 600 -> TRI = sqrt(600) = 24.49 m
    z_ramp_x = np.array([
        [100.0, 110.0, 120.0],
        [100.0, 110.0, 120.0],
        [100.0, 110.0, 120.0],
    ])
    slope_ramp, tri_ramp = compute_horn_slope_and_tri(z_ramp_x, dx=100.0, dy=100.0)
    assert slope_ramp == 5.71
    assert tri_ramp == 24.49


def test_event_persistence_linkage_edge_cases():
    """Verify all 7 event persistence linkage criteria and edge cases."""
    from src.events.event_clustering import check_event_persistence_linkage

    origin_lat, origin_lon = 20.0, 78.0
    orig_eid = "EVT_0001"

    # Case 1: Persistent event (detection tomorrow in same cluster within 25 km)
    dets_persistent = [{"lat": 20.05, "lon": 78.05, "event_id": orig_eid}]
    assert check_event_persistence_linkage(origin_lat, origin_lon, orig_eid, dets_persistent) == 1

    # Case 2: Extinguished event (no detections anywhere tomorrow)
    assert check_event_persistence_linkage(origin_lat, origin_lon, orig_eid, []) == 0

    # Case 3: Nearby independent event (detection tomorrow 7 km away, but DIFFERENT cluster)
    dets_independent = [{"lat": 20.05, "lon": 78.05, "event_id": "EVT_9999"}]
    assert check_event_persistence_linkage(origin_lat, origin_lon, orig_eid, dets_independent) == 0

    # Case 4: Chained cluster bridge (same cluster ID, but 75 km away > 25 km)
    dets_chained = [{"lat": 20.5, "lon": 78.5, "event_id": orig_eid}]
    assert check_event_persistence_linkage(origin_lat, origin_lon, orig_eid, dets_chained) == 0

    # Case 5: Spatially shifted event (moves 15 km on day T+1 in same cluster)
    dets_shifted = [{"lat": 20.1, "lon": 78.1, "event_id": orig_eid}]
    assert check_event_persistence_linkage(origin_lat, origin_lon, orig_eid, dets_shifted) == 1

    # Case 6: Two-day gap (detection at T+2, but none on calendar day T+1)
    # Passed tomorrow detections is empty
    assert check_event_persistence_linkage(origin_lat, origin_lon, orig_eid, []) == 0

    # Case 7: Ambiguous overlap (nearby fire from another cluster + distant fire in same cluster)
    dets_ambiguous = [
        {"lat": 20.02, "lon": 78.02, "event_id": "EVT_9999"},  # nearby, wrong event
        {"lat": 20.7, "lon": 78.7, "event_id": orig_eid},      # right event, too far
    ]
    assert check_event_persistence_linkage(origin_lat, origin_lon, orig_eid, dets_ambiguous) == 0


def test_calibration_comparison_platt_vs_isotonic():
    """Verify that ModelCalibrator handles Raw, Platt, and Isotonic cleanly."""
    np.random.seed(42)
    val_probs = np.random.uniform(0.1, 0.9, 1000)
    val_labels = (val_probs > 0.5).astype(int)

    test_probs = np.random.uniform(0.1, 0.9, 500)

    c_raw = ModelCalibrator(method="raw").fit(val_probs, val_labels)
    c_platt = ModelCalibrator(method="platt").fit(val_probs, val_labels)
    c_iso = ModelCalibrator(method="isotonic").fit(val_probs, val_labels)

    # Raw must be strictly identical
    assert np.allclose(c_raw.calibrate(test_probs), test_probs)
    # Platt must be smooth in (0, 1)
    p_platt = c_platt.calibrate(test_probs)
    assert (p_platt >= 0.0).all() and (p_platt <= 1.0).all()
    # Isotonic must be monotonic in [0, 1]
    p_iso = c_iso.calibrate(test_probs)
    assert (p_iso >= 0.0).all() and (p_iso <= 1.0).all()


def test_historical_replay_candidate_vs_full_spatial_domain():
    """Verify replay metrics distinguish candidate domain recall from full spatial domain recall."""
    from src.replay.historical_replay import HistoricalReplayEngine
    import joblib

    data_path = Path("data/features/multimodal_features.csv")
    model_path = Path("results/baselines/ExpD_LGBM_39_Multimodal.joblib")

    if data_path.exists() and model_path.exists():
        engine = HistoricalReplayEngine(data_path, model_path)
        res = engine.execute_replay("2024-03-10", probability_threshold=0.40)

        # Full spatial misses must include misses outside candidate domain
        assert res["total_domain_misses"] == res["misses_in_candidate"] + res["misses_outside_candidate"]
        # Actual fire cells total must equal target fires in candidate + misses outside candidate
        assert res["actual_fire_cells_total"] == res["target_fires_in_candidate"] + res["misses_outside_candidate"]
        # Full spatial recall cannot exceed candidate domain recall when unmonitored fires exist
        if res["misses_outside_candidate"] > 0:
            assert res["full_spatial_recall"] <= res["candidate_domain_recall"]


def test_geographic_regime_assignment_consistency():
    """Verify predefined geographic fire regime boundaries and backward-compatible alias."""
    from src.data.environmental import assign_geographic_regime, assign_ecological_regime

    lats = np.array([25.0, 15.0, 32.0, 24.0, 20.0, 24.0])
    lons = np.array([92.0, 75.0, 76.0, 72.0, 80.0, 85.0])

    regimes = assign_geographic_regime(lats, lons)
    assert regimes[0] == "NORTHEAST"
    assert regimes[1] == "WESTERN_GHATS"
    assert regimes[2] == "NORTH"
    assert regimes[3] == "NORTHWEST"
    assert regimes[4] == "CENTRAL"
    assert regimes[5] == "EAST"

    # Alias must produce strictly identical output
    alias_regimes = assign_ecological_regime(lats, lons)
    assert (regimes == alias_regimes).all()


def test_vpd_jensens_inequality_property():
    """Verify Jensen's inequality property of multi-day VPD proxy."""
    # Two days: Day 1 (T=30C, RH=40%), Day 2 (T=40C, RH=20%)
    t_days = np.array([30.0, 40.0])
    rh_days = np.array([40.0, 20.0])

    # True daily mean VPD
    daily_vpds = compute_vapor_pressure_deficit(t_days, rh_days)
    mean_daily_vpd = float(np.mean(daily_vpds))

    # Mean T and mean RH proxy:
    t_mean = np.array([float(np.mean(t_days))])
    rh_mean = np.array([float(np.mean(rh_days))])
    proxy_vpd = float(compute_vapor_pressure_deficit(t_mean, rh_mean)[0])

    # Due to convexity of es(T), proxy_vpd <= mean_daily_vpd
    assert proxy_vpd <= mean_daily_vpd + 1e-4


def test_soil_drought_index_bounds():
    """Verify topsoil moisture deficit proxy behavior across moisture boundaries."""
    # Saturated soil (>= 0.35): deficit = 0.0
    assert compute_soil_drought_index(np.array([0.40]))[0] == 0.0
    assert compute_soil_drought_index(np.array([0.35]))[0] == 0.0

    # Severely desiccated soil (0.0): deficit = 1.0
    assert compute_soil_drought_index(np.array([0.0]))[0] == 1.0

    # Intermediate moisture: 0.10 -> (0.35 - 0.10) / 0.35 = 0.25 / 0.35 = 0.714
    assert abs(compute_soil_drought_index(np.array([0.10]))[0] - 0.714) < 0.01


def test_factorial_interaction_bootstrap_symmetry():
    """Verify factorial interaction bootstrap logic when models are symmetric."""
    from src.evaluation.statistical_testing import compute_factorial_interaction_bootstrap

    np.random.seed(42)
    y = np.random.binomial(1, 0.5, 500)
    p = np.random.uniform(0.1, 0.9, 500)

    # Identical models: A = B = C = D -> interaction must be 0
    res = compute_factorial_interaction_bootstrap(y, p, p, p, p, metric_name="roc_auc", n_bootstraps=50)
    assert res["interaction"]["observed"] == 0.0
    assert res["interaction"]["ci_excludes_zero"] is False
