"""India Forest Fire Prediction & NASA FIRMS Surveillance Application.

Research-oriented Flask web application providing:
1. Live near-real-time satellite surveillance via NASA FIRMS VIIRS (375m)
   strictly filtered to sovereign India (Survey of India polygon boundary).
2. Occurrence risk classification using the validated 31-feature
   HistGradientBoostingClassifier model trained on ERA5-Land multi-timescale meteorology.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from firms_service import BOUNDARY_PATH, FIRMSService

logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "results" / "final_model" / "final_hgb_model.joblib"
METRICS_PATH = BASE_DIR / "results" / "final_model" / "metrics.json"

FEATURES = [
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


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact missing at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, metrics


MODEL, METRICS = load_artifacts()
firms_service = FIRMSService()


@app.route("/")
def index():
    """Main view presenting both Live Surveillance and 31-Feature Model inference."""
    has_key = bool(firms_service.get_api_key())
    active_tab = request.args.get("tab", "live")
    return render_template(
        "index.html",
        metrics=METRICS,
        has_key=has_key,
        active_tab=active_tab,
        input_values={},
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Predict forest fire occurrence using the 31-feature HistGradientBoosting model.
    
    Accepts both JSON payloads and HTML form submissions.
    Returns HTTP 400 on missing or invalid inputs.
    """
    result = None
    error = None
    input_values: Dict[str, Any] = {}

    if request.method == "POST":
        data = request.get_json(silent=True) if request.is_json else request.form
        if not data:
            error = "No input payload received"
            if request.is_json:
                return jsonify({"error": error}), 400
        else:
            try:
                for feature in FEATURES:
                    raw_val = data.get(feature)
                    if raw_val is None or str(raw_val).strip() == "":
                        raise ValueError(f"Missing required feature: '{feature}'")
                    try:
                        input_values[feature] = float(raw_val)
                    except (ValueError, TypeError):
                        raise ValueError(f"Feature '{feature}' must be a valid number, got: {raw_val}")

                row = pd.DataFrame([[input_values[f] for f in FEATURES]], columns=FEATURES)
                probability = float(MODEL.predict_proba(row)[0, 1])
                prediction = int(probability >= 0.5)
                result = {
                    "prediction": prediction,
                    "probability": probability * 100.0,
                    "label": "Fire Risk Detected" if prediction else "Low Fire Risk",
                }
                if request.is_json:
                    return jsonify(result)
            except Exception as exc:
                error = str(exc)
                if request.is_json:
                    return jsonify({"error": error}), 400

    has_key = bool(firms_service.get_api_key())
    status_code = 400 if (error and request.method == "POST") else 200
    return render_template(
        "index.html",
        result=result,
        error=error,
        metrics=METRICS,
        has_key=has_key,
        active_tab="model",
        input_values=input_values,
    ), status_code


# --- Real-Time NASA FIRMS Alert System API Endpoints ---

@app.route("/api/live-fires", methods=["GET"])
def api_live_fires():
    """Fetch active fire observations strictly filtered to sovereign India."""
    source = request.args.get("source", "ALL")
    day_range = int(request.args.get("days", 1))
    min_frp = float(request.args.get("min_frp", 0.0))
    min_confidence = request.args.get("confidence", "all")

    data = firms_service.fetch_live_fires(
        source=source,
        day_range=day_range,
        min_frp=min_frp,
        min_confidence=min_confidence,
    )
    return jsonify(data)


@app.route("/api/india-boundary", methods=["GET"])
def api_india_boundary():
    """Serve the official Survey of India boundary GeoJSON for client-side map rendering."""
    if not BOUNDARY_PATH.exists():
        return jsonify({"error": "India boundary file not found"}), 404
    return send_file(BOUNDARY_PATH, mimetype="application/geo+json")


@app.route("/api/firms-status", methods=["GET"])
def api_firms_status():
    """Report the current status of the NASA FIRMS configuration (safe, zero key exposure)."""
    has_key = bool(firms_service.get_api_key())
    return jsonify({
        "has_key": has_key,
        "mode": "LIVE" if has_key else "DEMO",
        "boundary_loaded": firms_service._prepared_india is not None,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
