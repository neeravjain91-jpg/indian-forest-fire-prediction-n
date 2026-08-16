from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURES = [
    "grid_lat",
    "grid_lon",
    "hour",
    "year",
    "month",
    "temp_1d",
    "rh_1d",
    "wind_1d",
    "pressure_1d",
    "soil_1d",
    "rain_1d",
    "temp_3d_mean",
    "temp_3d_max",
    "temp_3d_min",
    "rh_3d_mean",
    "rh_3d_min",
    "wind_3d_mean",
    "wind_3d_max",
    "pressure_3d_mean",
    "soil_3d_mean",
    "rain_3d_total",
    "temp_7d_mean",
    "temp_7d_max",
    "temp_7d_min",
    "rh_7d_mean",
    "rh_7d_min",
    "wind_7d_mean",
    "wind_7d_max",
    "pressure_7d_mean",
    "soil_7d_mean",
    "rain_7d_total",
]

TARGET = "fire"
MODEL_CONFIG = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test the final India forest-fire HGB model.")
    parser.add_argument("--data", required=True, help="Path to india_fire_weather_final.csv")
    parser.add_argument("--output", default="results/final_model", help="Directory for model artifacts")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["acq_date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Final model dataset contains missing values in required columns.")

    # Fixed temporal protocol used in the completed research experiments.
    train = df[df["year"] <= 2022].copy()
    validation = df[df["year"] == 2023].copy()
    test = df[df["year"] >= 2024].copy()

    X_train, y_train = train[FEATURES], train[TARGET].astype(int)
    X_val, y_val = validation[FEATURES], validation[TARGET].astype(int)
    X_test, y_test = test[FEATURES], test[TARGET].astype(int)

    model = HistGradientBoostingClassifier(**MODEL_CONFIG)
    model.fit(X_train, y_train)

    val_probability = model.predict_proba(X_val)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    results = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "date_min": str(df["acq_date"].min().date()),
            "date_max": str(df["acq_date"].max().date()),
            "fire_rows": int((df[TARGET] == 1).sum()),
            "nonfire_rows": int((df[TARGET] == 0).sum()),
        },
        "features": FEATURES,
        "model": MODEL_CONFIG,
        "split": {
            "train_years": "2018-2022",
            "validation_year": 2023,
            "test_years": "2024-2025",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        "validation": metrics(y_val, val_probability),
        "test": metrics(y_test, test_probability),
        "test_confusion_matrix": confusion_matrix(y_test, (test_probability >= 0.5).astype(int)).tolist(),
    }

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    importance.to_csv(outdir / "feature_importance.csv", index=False)
    joblib.dump(model, outdir / "final_hgb_model.joblib")

    pred = pd.DataFrame(
        {
            "grid_lat": test["grid_lat"].to_numpy(),
            "grid_lon": test["grid_lon"].to_numpy(),
            "acq_date": test["acq_date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "hour": test["hour"].to_numpy(),
            "actual_fire": y_test.to_numpy(),
            "fire_probability": test_probability,
            "predicted_fire": (test_probability >= 0.5).astype(int),
        }
    )
    pred.to_csv(outdir / "test_predictions.csv", index=False)

    (outdir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print("\nTOP 10 FEATURES")
    print(importance.head(10).to_string(index=False))
    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, (test_probability >= 0.5).astype(int), digits=4, zero_division=0))


if __name__ == "__main__":
    main()
