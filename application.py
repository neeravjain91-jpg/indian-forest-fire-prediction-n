from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = Path("results/final_model/final_hgb_model.joblib")
METRICS_PATH = Path("results/final_model/metrics.json")

FEATURES = [
    "grid_lat", "grid_lon", "hour", "year", "month",
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d",
    "temp_3d_mean", "temp_3d_max", "temp_3d_min", "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max", "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    "temp_7d_mean", "temp_7d_max", "temp_7d_min", "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max", "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found at {MODEL_PATH}. "
            "Run train_final_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return model, metrics


MODEL, METRICS = load_model()


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None

    if request.method == "POST":
        try:
            values = {feature: float(request.form[feature]) for feature in FEATURES}
            row = np.array([[values[feature] for feature in FEATURES]], dtype=float)
            probability = float(MODEL.predict_proba(row)[0, 1])
            prediction = int(probability >= 0.5)
            result = {
                "prediction": prediction,
                "probability": probability * 100.0,
                "label": "Fire Risk Detected" if prediction else "Low Fire Risk",
            }
        except (KeyError, ValueError, TypeError) as exc:
            error = f"Invalid input: {exc}"

    return render_template("index.html", result=result, error=error, metrics=METRICS)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
