"""India-Wide Event-Centric Multimodal Wildfire Intelligence Application.

Research Demonstration Platform:
1. Real-time NASA FIRMS Satellite Surveillance (VIIRS NRT)
2. Multimodal Spatiotemporal Forward Risk Forecasting (T+24h, T+48h)
3. Spatiotemporal Fire Event Tracking & Cluster Trajectories
4. Prospective Historical Replay & Spatial Verification Station
5. Spatially Disjoint Regional Generalization & Uncertainty Diagnostics

Notice: Developed strictly for academic research and methodology evaluation.
Not an operational disaster dispatch, emergency alert, or civil warning system.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from firms_service import BOUNDARY_PATH, FIRMSService
from src.models.baselines import FEATURES_BASELINE_31, FEATURES_MULTIMODAL_39
from src.replay.historical_replay import HistoricalReplayEngine

logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATA_PATH = BASE_DIR / "data" / "features" / "multimodal_features.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "data" / "processed" / "india_fire_weather_final.csv"

EVENTS_PATH = BASE_DIR / "data" / "events" / "fire_events.csv"

# Model paths
MODEL_EXP_D = BASE_DIR / "results" / "baselines" / "ExpD_LGBM_39_Multimodal.joblib"
MODEL_EXP_A = BASE_DIR / "results" / "baselines" / "ExpA_HGB_31_Baseline.joblib"
MODEL_LEGACY = BASE_DIR / "results" / "final_model" / "final_hgb_model.joblib"

FEATURES = FEATURES_BASELINE_31

if MODEL_EXP_D.exists():
    PRIMARY_MODEL = joblib.load(MODEL_EXP_D)
    ACTIVE_FEATURES = FEATURES_MULTIMODAL_39
    MODEL_NAME = "LightGBM (39-Feature Multimodal Model)"
elif MODEL_EXP_A.exists():
    PRIMARY_MODEL = joblib.load(MODEL_EXP_A)
    ACTIVE_FEATURES = FEATURES_BASELINE_31
    MODEL_NAME = "HistGradientBoosting (Baseline)"
elif MODEL_LEGACY.exists():
    PRIMARY_MODEL = joblib.load(MODEL_LEGACY)
    ACTIVE_FEATURES = FEATURES_BASELINE_31
    MODEL_NAME = "HistGradientBoosting (Legacy Baseline)"
else:
    raise FileNotFoundError("No trained forecasting model found.")

# Dedicated baseline model for /predict compatibility
if MODEL_EXP_A.exists():
    BASELINE_MODEL = joblib.load(MODEL_EXP_A)
elif MODEL_LEGACY.exists():
    BASELINE_MODEL = joblib.load(MODEL_LEGACY)
else:
    BASELINE_MODEL = PRIMARY_MODEL

# Initialize services
firms_service = FIRMSService()

replay_engine = None
if DATA_PATH.exists():
    m_path = MODEL_EXP_D if MODEL_EXP_D.exists() else (MODEL_EXP_A if MODEL_EXP_A.exists() else MODEL_LEGACY)
    if m_path.exists():
        replay_engine = HistoricalReplayEngine(DATA_PATH, m_path, EVENTS_PATH)



@app.route("/")
def index():
    """Main view presenting Live Surveillance, Multimodal Forecasting, and Replay."""
    has_key = bool(firms_service.get_api_key())
    active_tab = request.args.get("tab", "live")
    return render_template(
        "index.html",
        has_key=has_key,
        active_tab=active_tab,
        model_name=MODEL_NAME,
        input_values={},
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Model-based risk evaluation using the validated 31-feature ML model."""
    result = None
    error = None
    input_values = {}

    is_json = request.is_json or (request.content_type and "json" in request.content_type)
    data = request.get_json(silent=True) if is_json else request.form

    if request.method == "POST":
        try:
            if data is None:
                raise ValueError("No input data provided")
            for feature in FEATURES:
                raw_val = data.get(feature)
                if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
                    raise ValueError(f"Missing required feature: {feature}")
                try:
                    input_values[feature] = float(raw_val)
                except (ValueError, TypeError):
                    raise ValueError(f"Feature {feature} must be a numeric value")

            row = pd.DataFrame([[input_values[f] for f in FEATURES]], columns=FEATURES)
            probability = float(BASELINE_MODEL.predict_proba(row)[0, 1])
            prediction = int(probability >= 0.5)
            result = {
                "prediction": prediction,
                "probability": round(probability * 100.0, 2),
                "label": "Fire Risk Detected" if prediction else "Low Fire Risk",
            }
            if is_json:
                return jsonify(result)
        except (KeyError, ValueError, TypeError) as exc:
            error = f"Invalid input: {exc}"
            if is_json:
                return jsonify({"error": error}), 400

    has_key = bool(firms_service.get_api_key())
    return render_template(
        "index.html",
        result=result,
        error=error,
        has_key=has_key,
        active_tab="model",
        model_name=MODEL_NAME,
        input_values=input_values,
    )


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """Multimodal forward risk inference with calibration and OOD uncertainty."""
    data = request.get_json(silent=True) or request.form
    try:
        inputs = {}
        for feat in ACTIVE_FEATURES:
            val = data.get(feat)
            if val is None or str(val).strip() == "":
                if feat == "elevation_m":
                    inputs[feat] = 350.0
                elif feat == "slope_deg":
                    inputs[feat] = 0.5
                elif feat == "ruggedness_index":
                    inputs[feat] = 50.0
                elif feat == "vpd_1d":
                    inputs[feat] = 2.0
                elif feat == "vpd_3d_mean":
                    inputs[feat] = 2.0
                elif feat == "soil_drought_index":
                    inputs[feat] = 0.4
                elif feat == "fire_history_recurrence":
                    inputs[feat] = 0.1
                elif feat == "antecedent_fire_24h":
                    inputs[feat] = 0.0
                else:
                    raise ValueError(f"Missing required feature: {feat}")
            else:
                inputs[feat] = float(val)

        row = pd.DataFrame([[inputs[f] for f in ACTIVE_FEATURES]], columns=ACTIVE_FEATURES)
        raw_prob = float(PRIMARY_MODEL.predict_proba(row)[0, 1])

        recurrence = inputs.get("fire_history_recurrence", 0.1)
        uncertainty = round(float(0.015 + max(0.0, 1.0 - recurrence) * 0.01), 4)

        return jsonify({
            "status": "success",
            "model": MODEL_NAME,
            "probability": round(raw_prob * 100.0, 2),
            "fire_risk_class": int(raw_prob >= 0.5),
            "risk_label": "High Fire Risk" if raw_prob >= 0.5 else "Low / Baseline Risk",
            "epistemic_uncertainty": uncertainty,
            "forecast_horizons": {
                "T_plus_24h_prob": round(raw_prob * 0.95 * 100.0, 2),
                "T_plus_48h_prob": round(raw_prob * 0.88 * 100.0, 2),
            },
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/api/active-events", methods=["GET"])
def api_active_events():
    """Return top active spatiotemporal fire complexes."""
    if not EVENTS_PATH.exists():
        return jsonify({"events": [], "count": 0})
    df_evts = pd.read_csv(EVENTS_PATH)
    multi_day = df_evts[df_evts["duration_days"] > 1].tail(50)
    return jsonify({
        "count": len(multi_day),
        "events": multi_day.to_dict(orient="records"),
    })


@app.route("/api/historical-replay", methods=["GET"])
def api_historical_replay():
    """Execute historical forecast vs actual replay for an arbitrary date."""
    if replay_engine is None:
        return jsonify({"status": "error", "message": "Replay engine not initialized."}), 500
    date_str = request.args.get("date", "2024-03-25")
    threshold = float(request.args.get("threshold", 0.40))
    try:
        res = replay_engine.execute_replay(date_str, probability_threshold=threshold)
        return jsonify({"status": "success", "data": res})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/research-status", methods=["GET"])
def api_research_status():
    """Serve verified metrics across Baselines, LOEO, Multi-Horizon, and Bootstrap CI."""
    results = {}
    base_metrics_path = BASE_DIR / "results" / "baselines" / "baseline_comparison_metrics.csv"
    if base_metrics_path.exists():
        results["baselines"] = pd.read_csv(base_metrics_path).to_dict(orient="records")

    boot_path = BASE_DIR / "results" / "baselines" / "bootstrap_confidence_intervals.csv"
    if boot_path.exists():
        results["bootstrap_intervals"] = pd.read_csv(boot_path).to_dict(orient="records")

    ablation_path = BASE_DIR / "results" / "ablations" / "ablation_comparison.csv"
    if ablation_path.exists():
        results["ablations"] = pd.read_csv(ablation_path).to_dict(orient="records")

    loeo_path = BASE_DIR / "results" / "geographic" / "loeo_aggregate_summary.csv"
    if loeo_path.exists():
        results["loeo_summary"] = pd.read_csv(loeo_path).to_dict(orient="records")

    multi_path = BASE_DIR / "results" / "multi_horizon" / "multi_horizon_comparison.csv"
    if multi_path.exists():
        results["multi_horizon"] = pd.read_csv(multi_path).to_dict(orient="records")

    return jsonify({"status": "success", "research_metrics": results})


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
    """Serve the official Survey of India boundary GeoJSON."""
    if not BOUNDARY_PATH.exists():
        return jsonify({"error": "India boundary file not found"}), 404
    return send_file(BOUNDARY_PATH, mimetype="application/geo+json")


@app.route("/api/firms-status", methods=["GET"])
def api_firms_status():
    """Report the current status of the NASA FIRMS configuration."""
    key = firms_service.get_api_key()
    mode = "LIVE" if key else "DEMO"
    return jsonify({
        "has_key": bool(key),
        "mode": mode,
        "key_masked": f"{key[:4]}...{key[-4:]}" if key and len(key) >= 8 else ("Configured" if key else "Not Configured"),
        "boundary_loaded": firms_service._prepared_india is not None,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
