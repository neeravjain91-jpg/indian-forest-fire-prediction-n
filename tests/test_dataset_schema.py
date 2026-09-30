"""Tests for dataset integrity, schema, and spatiotemporal bounds.

Validates that data/processed/india_fire_weather_final.csv adheres strictly to
the expected research schema:
- 131,000 observations
- Balanced fire (65,518) vs non-fire (65,482)
- Exactly 31 predictive features
- Spatial extent and cell counts
- Chronological span (2018-2025)
- Zero nulls across all predictive features
"""
from pathlib import Path
import pandas as pd
import pytest

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "india_fire_weather_final.csv"

EXPECTED_FEATURES = [
    # Spatial (2)
    "grid_lat", "grid_lon",
    # Temporal (3)
    "hour", "year", "month",
    # 1-day Weather (6)
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d",
    # 3-day Weather (10)
    "temp_3d_mean", "temp_3d_max", "temp_3d_min",
    "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max",
    "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    # 7-day Weather (10)
    "temp_7d_mean", "temp_7d_max", "temp_7d_min",
    "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max",
    "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]


@pytest.fixture(scope="module")
def dataset_df():
    """Load the final validated research dataset."""
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH)
    return df


def test_dataset_file_exists():
    """Dataset file must exist in data/processed/."""
    assert DATASET_PATH.exists()


def test_dataset_row_and_column_count(dataset_df):
    """Dataset must contain exactly 131,000 observations and 38 columns."""
    assert len(dataset_df) == 131000
    assert dataset_df.shape[1] == 38


def test_predictive_feature_schema(dataset_df):
    """All 31 predictive features must be present in the dataset."""
    assert len(EXPECTED_FEATURES) == 31
    for feature in EXPECTED_FEATURES:
        assert feature in dataset_df.columns, f"Missing feature: {feature}"


def test_no_null_values_in_predictive_features(dataset_df):
    """No predictive feature may contain NaN or null values."""
    null_counts = dataset_df[EXPECTED_FEATURES].isnull().sum()
    assert null_counts.sum() == 0, f"Found nulls in features:\n{null_counts[null_counts > 0]}"


def test_class_balance(dataset_df):
    """Dataset must have 65,518 fire (1) and 65,482 non-fire (0) observations."""
    assert "fire" in dataset_df.columns
    counts = dataset_df["fire"].value_counts().to_dict()
    assert counts.get(1) == 65518
    assert counts.get(0) == 65482


def test_temporal_span(dataset_df):
    """Dataset must span chronological years 2018 through 2025."""
    years = sorted(dataset_df["year"].unique().tolist())
    assert years == [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def test_spatial_coverage(dataset_df):
    """Grid latitude and longitude must fall within the India bounding region."""
    assert dataset_df["grid_lat"].min() >= 6.0
    assert dataset_df["grid_lat"].max() <= 38.0
    assert dataset_df["grid_lon"].min() >= 68.0
    assert dataset_df["grid_lon"].max() <= 98.0
    
    unique_cells = len(dataset_df.groupby(["grid_lat", "grid_lon"]))
    assert unique_cells == 26494
