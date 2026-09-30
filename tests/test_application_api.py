"""Tests for Flask application endpoints, JSON API, form submissions, and input validation."""
import json
import pytest
from application import app, FEATURES


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_payload():
    """Construct a valid 31-feature input payload."""
    payload = {
        "grid_lat": 21.8, "grid_lon": 86.3, "hour": 8, "year": 2024, "month": 4,
        "temp_1d": 39.5, "rh_1d": 18.0, "wind_1d": 4.8, "pressure_1d": 1003.0, "soil_1d": 0.08, "rain_1d": 0.0,
        "temp_3d_mean": 38.6, "temp_3d_max": 42.1, "temp_3d_min": 27.0,
        "rh_3d_mean": 20.5, "rh_3d_min": 14.0,
        "wind_3d_mean": 4.2, "wind_3d_max": 7.5,
        "pressure_3d_mean": 1004.2, "soil_3d_mean": 0.09, "rain_3d_total": 0.0,
        "temp_7d_mean": 37.8, "temp_7d_max": 43.0, "temp_7d_min": 25.5,
        "rh_7d_mean": 23.0, "rh_7d_min": 12.0,
        "wind_7d_mean": 3.9, "wind_7d_max": 8.1,
        "pressure_7d_mean": 1005.0, "soil_7d_mean": 0.10, "rain_7d_total": 0.0,
    }
    assert len(payload) == 31
    return payload


def test_index_page(client):
    """GET / must render the dashboard successfully with 200 OK."""
    resp = client.get("/")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert "India Forest Fire" in text
    assert "Risk Classifier" in text
    assert "Active Surveillance" in text


def test_api_firms_status(client):
    """GET /api/firms-status must report operational status without leaking credentials."""
    resp = client.get("/api/firms-status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "has_key" in data
    assert data["mode"] in ("LIVE", "DEMO")
    assert data["boundary_loaded"] is True
    # Ensure no actual key value is returned
    assert "FIRMS_MAP_KEY" not in data
    assert "key" not in data or isinstance(data.get("key"), bool)


def test_api_india_boundary(client):
    """GET /api/india-boundary must serve valid GeoJSON with correct mime type."""
    resp = client.get("/api/india-boundary")
    assert resp.status_code == 200
    assert "json" in resp.content_type
    geojson = resp.get_json()
    assert geojson.get("type") in ("FeatureCollection", "Feature", "Polygon", "MultiPolygon")


def test_api_live_fires(client):
    """GET /api/live-fires must return detection list and summary statistics."""
    resp = client.get("/api/live-fires?days=1&min_frp=0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data
    assert "summary" in data
    assert "fires" in data
    assert "total_fires" in data["summary"]
    assert "severe_count" in data["summary"]


def test_predict_endpoint_valid_json(client, valid_payload):
    """POST /predict with valid 31 features via JSON returns prediction result."""
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 100.0
    assert "label" in data


def test_predict_endpoint_missing_feature(client, valid_payload):
    """POST /predict missing a required feature returns HTTP 400."""
    incomplete = dict(valid_payload)
    del incomplete["temp_1d"]
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_predict_endpoint_invalid_type(client, valid_payload):
    """POST /predict with non-numeric value returns HTTP 400."""
    invalid = dict(valid_payload)
    invalid["temp_1d"] = "not-a-number"
    resp = client.post("/predict", json=invalid)
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_predict_endpoint_valid_form(client, valid_payload):
    """POST /predict via HTML form submission returns 200 rendered template."""
    resp = client.post("/predict", data=valid_payload)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Prediction Result" in html or "Risk" in html


def test_api_forecast_multimodal(client, valid_payload):
    """POST /api/forecast returns multi-horizon probability, risk class, and epistemic uncertainty."""
    resp = client.post("/api/forecast", json=valid_payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 100.0
    assert "epistemic_uncertainty" in data
    assert "forecast_horizons" in data
    assert "T_plus_24h_prob" in data["forecast_horizons"]
    assert "T_plus_48h_prob" in data["forecast_horizons"]


def test_api_active_events(client):
    """GET /api/active-events returns tracked multi-day wildfire event complexes."""
    resp = client.get("/api/active-events")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "count" in data
    assert "events" in data
    assert isinstance(data["events"], list)


def test_api_historical_replay(client):
    """GET /api/historical-replay evaluates prospective spatial prediction against ground truth."""
    resp = client.get("/api/historical-replay?date=2024-03-25&threshold=0.40")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert "data" in data
    assert "candidate_domain_recall" in data["data"]
    assert "full_spatial_recall" in data["data"]



def test_api_research_status(client):
    """GET /api/research-status serves scientific benchmark metrics across all research suites."""
    resp = client.get("/api/research-status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    metrics = data["research_metrics"]
    assert "baselines" in metrics
    assert "bootstrap_intervals" in metrics

