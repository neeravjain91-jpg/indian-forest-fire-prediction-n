"""Tests for India spatial boundary integrity, GeoJSON structure, and geometry validation."""
import json
from pathlib import Path
import pytest
from shapely.geometry import Point, shape

from firms_service import FIRMSService

BASE_DIR = Path(__file__).resolve().parent.parent
BOUNDARY_PATH = BASE_DIR / "data" / "processed" / "india_boundary.geojson"


def test_boundary_file_exists_and_parses():
    """Boundary file must exist and contain valid GeoJSON geometry."""
    assert BOUNDARY_PATH.exists(), f"Boundary missing at {BOUNDARY_PATH}"
    with open(BOUNDARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("type") in ("FeatureCollection", "Feature", "Polygon", "MultiPolygon")
    if data.get("type") == "FeatureCollection":
        assert len(data.get("features", [])) > 0
        geom = shape(data["features"][0]["geometry"])
    elif "geometry" in data:
        geom = shape(data["geometry"])
    else:
        geom = shape(data)

    assert geom.is_valid or geom.buffer(0).is_valid
    bounds = geom.bounds  # minx, miny, maxx, maxy
    assert 65.0 <= bounds[0] <= 70.0  # West longitude
    assert 5.0 <= bounds[1] <= 10.0   # South latitude
    assert 95.0 <= bounds[2] <= 100.0 # East longitude
    assert 34.0 <= bounds[3] <= 38.0  # North latitude


def test_inside_india_points():
    """Known coordinates inside sovereign India must evaluate to True."""
    service = FIRMSService(boundary_path=BOUNDARY_PATH)
    
    inside_points = [
        (77.2090, 28.6139, "New Delhi"),
        (86.3218, 21.8542, "Similipal National Park, Odisha"),
        (77.5946, 12.9716, "Bengaluru, Karnataka"),
        (72.8777, 19.0760, "Mumbai, Maharashtra"),
        (78.4510, 22.3420, "Satpura Tiger Reserve, MP"),
        (92.7176, 26.2006, "Assam, Northeast"),
    ]
    for lon, lat, name in inside_points:
        assert service.is_inside_india(lon, lat), f"Expected {name} ({lon}, {lat}) to be inside India"


def test_outside_india_points():
    """Coordinates clearly outside sovereign India must evaluate to False."""
    service = FIRMSService(boundary_path=BOUNDARY_PATH)
    
    outside_points = [
        (-0.1276, 51.5074, "London, UK"),
        (79.8612, 6.9271, "Colombo, Sri Lanka"),
        (90.4125, 23.8103, "Dhaka, Bangladesh"),
        (67.0011, 24.8607, "Karachi, Pakistan"),
        (60.0000, 15.0000, "Arabian Sea international waters"),
        (90.0000, 5.0000, "Southern Indian Ocean"),
    ]
    for lon, lat, name in outside_points:
        assert not service.is_inside_india(lon, lat), f"Expected {name} ({lon}, {lat}) to be outside India"


def test_missing_boundary_raises_runtime_error(tmp_path):
    """If the boundary polygon file is missing or uninitialized, is_inside_india must raise RuntimeError."""
    non_existent = tmp_path / "does_not_exist.geojson"
    service = FIRMSService(boundary_path=non_existent)
    assert service._prepared_india is None

    with pytest.raises(RuntimeError, match="India sovereign boundary polygon is not loaded"):
        service.is_inside_india(77.2, 28.6)
