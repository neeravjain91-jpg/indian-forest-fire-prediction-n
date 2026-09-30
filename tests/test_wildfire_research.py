"""Scientific verification test suite for event-centric wildfire intelligence framework."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.environmental import (
    assign_ecological_regime,
    compute_soil_drought_index,
    compute_vapor_pressure_deficit,
)
from src.data.terrain import compute_terrain_features
from src.evaluation.calibration import ModelCalibrator, UncertaintyEstimator
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_expected_calibration_error,
)
from src.events.event_clustering import (
    cluster_fire_events,
    compute_bearing_deg,
    haversine_km,
)


def test_terrain_computation_bounds():
    """Verify terrain features return physically plausible geomorphology for India."""
    # Test coordinates: Himalayas (North), Western Ghats (West), Deccan (Central), Plains
    lats = np.array([32.0, 14.5, 22.0, 26.0])
    lons = np.array([77.0, 74.5, 78.0, 82.0])
    terrain = compute_terrain_features(lats, lons)

    assert len(terrain) == 4
    assert (terrain["elevation_m"] >= 10.0).all()
    assert (terrain["slope_deg"] >= 0.5).all()
    assert (terrain["ruggedness_index"] >= 1.0).all()
    # Himalayan point elevation should exceed plains elevation
    assert terrain.loc[0, "elevation_m"] > terrain.loc[3, "elevation_m"]
    # Western Ghats slope should exceed alluvial plains slope
    assert terrain.loc[1, "slope_deg"] > terrain.loc[3, "slope_deg"]


def test_environmental_and_vpd_physics():
    """Verify VPD and drought indices adhere to atmospheric thermodynamics."""
    temp_c = np.array([20.0, 35.0, 35.0])
    rh_pct = np.array([80.0, 80.0, 20.0])  # Hot & dry should yield highest VPD

    vpd = compute_vapor_pressure_deficit(temp_c, rh_pct)
    assert (vpd >= 0.0).all()
    # Hot and dry (index 2) must have higher VPD than hot and humid (index 1)
    assert vpd[2] > vpd[1]
    # Warmer air holds more moisture, so 35C at 80% RH has higher VPD than 20C at 80% RH
    assert vpd[1] > vpd[0]

    # Drought index
    soil_moisture = np.array([0.10, 0.25, 0.40])
    drought = compute_soil_drought_index(soil_moisture)
    assert drought[0] > drought[1] >= drought[2]


def test_ecological_regime_assignment():
    """Verify coordinates are partitioned into correct ecological biomes."""
    lats = np.array([30.5, 25.0, 13.0, 23.0, 24.0, 26.0])
    lons = np.array([78.0, 92.5, 75.0, 79.0, 85.0, 72.0])
    regimes = assign_ecological_regime(lats, lons)

    assert regimes[0] == "NORTH"           # Himalayas / Uttarakhand
    assert regimes[1] == "NORTHEAST"       # Assam / Purvanchal
    assert regimes[2] == "WESTERN_GHATS"   # Karnataka / Western Ghats
    assert regimes[3] == "CENTRAL"         # MP / Deccan
    assert regimes[4] == "EAST"            # Jharkhand / Eastern Ghats
    assert regimes[5] == "NORTHWEST"       # Rajasthan / Aravalli


def test_spatiotemporal_event_clustering():
    """Verify active fire clustering accurately groups adjacent detections in space and time."""
    test_detections = pd.DataFrame({
        "grid_lat": [20.0, 20.1, 20.2, 28.0],
        "grid_lon": [78.0, 78.1, 78.1, 85.0],
        "acq_date": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-01"]),
        "max_frp": [10.5, 25.0, 8.0, 5.0],
        "fire_detections": [1, 2, 1, 1],
    })

    clustered, summary = cluster_fire_events(
        test_detections, spatial_radius_km=30.0, temporal_gap_days=2
    )

    # First 3 points are close in space and consecutive in time -> 1 event
    # 4th point is 900+ km away in space -> distinct event
    assert len(summary) == 2
    assert summary.iloc[0]["detection_count"] == 3
    assert summary.iloc[0]["duration_days"] == 3
    assert summary.iloc[0]["max_frp"] == 25.0


def test_haversine_and_bearing():
    """Verify great circle distance and compass azimuth calculations."""
    # Distance between (0, 0) and (0, 1) deg longitude at equator is ~111.19 km
    dist = haversine_km(0.0, 0.0, 0.0, 1.0)
    assert 111.0 <= dist <= 111.5

    # Due east bearing should be 90 degrees
    bearing = compute_bearing_deg(0.0, 0.0, 0.0, 1.0)
    assert abs(bearing - 90.0) < 0.1


def test_calibration_and_ece():
    """Verify probability calibration shrinks Expected Calibration Error (ECE)."""
    np.random.seed(42)
    # Simulate uncalibrated overconfident probabilities
    y_true = np.random.binomial(1, 0.4, size=1000)
    raw_probs = np.clip(y_true * 0.8 + np.random.uniform(0, 0.4, size=1000), 0.05, 0.95)

    initial_ece = compute_expected_calibration_error(y_true, raw_probs)

    calibrator = ModelCalibrator(method="isotonic")
    calibrator.fit(raw_probs[:500], y_true[:500])
    cal_probs = calibrator.calibrate(raw_probs[500:])

    cal_ece = compute_expected_calibration_error(y_true[500:], cal_probs)
    assert cal_ece <= initial_ece


def test_uncertainty_estimator():
    """Verify Out-Of-Distribution (OOD) distance metric calculation."""
    X_train = np.random.randn(500, 10)
    X_test_in = np.random.randn(50, 10)  # In-distribution
    X_test_out = np.random.randn(50, 10) + 10.0  # Far out-of-distribution

    estimator = UncertaintyEstimator().fit(X_train)
    ood_in = estimator.compute_ood_distance(X_test_in)
    ood_out = estimator.compute_ood_distance(X_test_out)

    assert (ood_in >= 0.0).all() and (ood_in <= 1.0).all()
    assert (ood_out >= 0.0).all() and (ood_out <= 1.0).all()
    # Out-of-distribution samples must have substantially higher OOD score
    assert np.mean(ood_out) > np.mean(ood_in)


def test_data_leakage_chronological_splits():
    """Verify strict chronological separation between train, val, and test splits."""
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


def test_spatial_split_disjointness():
    """Verify complete geographic disjointness between spatial holdout sets."""
    train_path = Path("data/splits/train_spatial_non_central.csv")
    test_path = Path("data/splits/test_spatial_central.csv")

    if train_path.exists() and test_path.exists():
        tr = pd.read_csv(train_path, usecols=["ecological_regime"])
        te = pd.read_csv(test_path, usecols=["ecological_regime"])

        assert "CENTRAL" not in tr["ecological_regime"].values
        assert (te["ecological_regime"] == "CENTRAL").all()
