from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from firms_service import BOUNDARY_PATH, FIRMSService

app = Flask(__name__)

# Paths to the validated, immutable research artifacts
MODEL_PATH = Path("results/final_model/final_hgb_model.joblib")
METRICS_PATH = Path("results/final_model/metrics.json")

# Validated 31 features used by the HistGradientBoostingClassifier model
FEATURES = [
    "grid_lat", "grid_lon", "hour", "year", "month",
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d",
    "temp_3d_mean", "temp_3d_max", "temp_3d_min", "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max", "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    "temp_7d_mean", "temp_7d_max", "temp_7d_min", "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max", "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]


def load_model():
    """Load the final validated research model and metrics."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found at {MODEL_PATH}. "
            "Research model artifact must be present."
        )
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return model, metrics


MODEL, METRICS = load_model()
firms_service = FIRMSService()


@app.route("/")
def index():
    """Main view presenting both Live Fire Detection and Research Model capabilities."""
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
    """Model-based risk evaluation using the validated 31-feature ML model."""
    result = None
    error = None
    input_values = {}

    if request.method == "POST":
        try:
            for feature in FEATURES:
                raw_val = request.form.get(feature)
                if raw_val is None or raw_val.strip() == "":
                    raise ValueError(f"Missing required feature: {feature}")
                input_values[feature] = float(raw_val)

            row = pd.DataFrame([[input_values[f] for f in FEATURES]], columns=FEATURES)
            probability = float(MODEL.predict_proba(row)[0, 1])
            prediction = int(probability >= 0.5)
            result = {
                "prediction": prediction,
                "probability": probability * 100.0,
                "label": "Fire Risk Detected" if prediction else "Low Fire Risk",
            }
        except (KeyError, ValueError, TypeError) as exc:
            error = f"Invalid input: {exc}"

    has_key = bool(firms_service.get_api_key())
    return render_template(
        "index.html",
        result=result,
        error=error,
        metrics=METRICS,
        has_key=has_key,
        active_tab="model",
        input_values=input_values,
    )


# --- Real-Time NASA FIRMS Alert System API Endpoints ---

@app.route("/api/live-fires", methods=["GET"])
def api_live_fires():
    """Fetch real-time active fire observations from NASA FIRMS strictly filtered to India."""
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
    """Report the current status of the NASA FIRMS configuration."""
    key = firms_service.get_api_key()
    return jsonify({
        "has_key": bool(key),
        "key_masked": f"{key[:4]}...{key[-4:]}" if key and len(key) >= 8 else ("Configured" if key else "Not Configured"),
        "boundary_loaded": firms_service._prepared_india is not None,
    })


@app.route("/api/set-key", methods=["POST"])
def api_set_key():
    """Allow setting FIRMS_MAP_KEY for the current session without saving to disk or git."""
    data = request.get_json(silent=True) or request.form
    key = str(data.get("key", "")).strip()
    if key:
        os.environ["FIRMS_MAP_KEY"] = key
        # Invalidate cache so fresh live data is fetched
        firms_service._cache.clear()
        return jsonify({"status": "success", "message": "FIRMS MAP_KEY registered in session."})
    return jsonify({"status": "error", "message": "No key provided"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
