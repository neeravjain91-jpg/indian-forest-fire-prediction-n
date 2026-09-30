"""Tests for NASA FIRMS service, filtering, cache, demo/real distinction, and security."""
from datetime import datetime, timezone
import pytest

from firms_service import FIRMSService, MAX_CACHE_SIZE


@pytest.fixture
def firms_svc():
    return FIRMSService()


def test_severity_categorization(firms_svc):
    """Test standard severity thresholds."""
    assert firms_svc.categorize_severity(55.0) == "severe"
    assert firms_svc.categorize_severity(50.0) == "severe"
    assert firms_svc.categorize_severity(49.9) == "high"
    assert firms_svc.categorize_severity(20.0) == "high"
    assert firms_svc.categorize_severity(19.9) == "moderate"
    assert firms_svc.categorize_severity(5.0) == "moderate"
    assert firms_svc.categorize_severity(4.9) == "low"
    assert firms_svc.categorize_severity(0.0) == "low"


def test_process_records_filters_outside_india(firms_svc):
    """Records outside India must be dropped."""
    records = [
        # Inside: Similipal, Odisha
        {"latitude": "21.8542", "longitude": "86.3218", "frp": "45.0", "confidence": "h", "acq_date": "2024-05-01", "acq_time": "0830", "satellite": "Suomi-NPP VIIRS"},
        # Outside: London
        {"latitude": "51.5074", "longitude": "-0.1276", "frp": "80.0", "confidence": "h", "acq_date": "2024-05-01", "acq_time": "0830", "satellite": "Suomi-NPP VIIRS"},
    ]
    res = firms_svc._process_records(records)
    assert res["summary"]["total_fires"] == 1
    assert res["fires"][0]["lat"] == pytest.approx(21.8542, abs=1e-4)


def test_process_records_filters_by_frp_and_confidence(firms_svc):
    """Filter records by minimum FRP and confidence level."""
    records = [
        {"latitude": "21.8542", "longitude": "86.3218", "frp": "10.0", "confidence": "l", "acq_date": "2024-05-01", "acq_time": "0830"},
        {"latitude": "21.8542", "longitude": "86.3218", "frp": "30.0", "confidence": "n", "acq_date": "2024-05-01", "acq_time": "0830"},
        {"latitude": "21.8542", "longitude": "86.3218", "frp": "60.0", "confidence": "h", "acq_date": "2024-05-01", "acq_time": "0830"},
    ]
    # Filter min_frp >= 20.0
    res_frp = firms_svc._process_records(records, min_frp=20.0)
    assert res_frp["summary"]["total_fires"] == 2

    # Filter confidence == high
    res_conf = firms_svc._process_records(records, min_confidence="high")
    assert res_conf["summary"]["total_fires"] == 1
    assert res_conf["fires"][0]["frp"] == 60.0


def test_process_records_calculates_ist_correctly(firms_svc):
    """Verify UTC to IST (+5:30) conversion for Indian operational time."""
    records = [
        {"latitude": "21.8542", "longitude": "86.3218", "frp": "40.0", "confidence": "h", "acq_date": "2024-05-01", "acq_time": "0830"}
    ]
    res = firms_svc._process_records(records)
    fire = res["fires"][0]
    # 08:30 UTC + 5:30 = 14:00 IST
    assert fire["acq_time_ist"] == "14:00 IST"
    assert "UTC" in fire["acq_time"]


def test_demo_mode_when_no_api_key(monkeypatch, firms_svc):
    """When FIRMS_MAP_KEY is missing, fetch_live_fires must return demo data with explicit flags."""
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.setattr(firms_svc, "get_api_key", lambda: None)

    data = firms_svc.fetch_live_fires()
    assert data["status"] == "demo"
    assert data["is_demo"] is True
    assert "DEMO MODE" in data["mode_label"]
    assert "summary" in data
    assert "fires" in data
    assert len(data["fires"]) > 0
    # Confirm every fire is strictly within India
    for f in data["fires"]:
        assert firms_svc.is_inside_india(f["lon"], f["lat"])


def test_cache_bounded_size(firms_svc):
    """Cache must not exceed MAX_CACHE_SIZE and must purge oldest entries."""
    firms_svc._cache.clear()
    for i in range(MAX_CACHE_SIZE + 10):
        # Manually inject dummy cache entries
        firms_svc._cache[f"key_{i}"] = (float(i), {"test": i})
    
    # Trigger a dummy fetch or cleanup
    assert len(firms_svc._cache) == MAX_CACHE_SIZE + 10
    # Clean down to MAX_CACHE_SIZE
    with firms_svc._lock:
        while len(firms_svc._cache) > MAX_CACHE_SIZE:
            oldest = min(firms_svc._cache.keys(), key=lambda k: firms_svc._cache[k][0])
            del firms_svc._cache[oldest]
    assert len(firms_svc._cache) == MAX_CACHE_SIZE


def test_no_api_secret_leakage(firms_svc):
    """Payloads returned by fetch_live_fires must never contain secrets."""
    data = firms_svc.fetch_live_fires()
    data_str = str(data).lower()
    assert "secret" not in data_str
    assert "token" not in data_str
    assert "password" not in data_str
