"""Train and evaluate the primary HistGradientBoostingClassifier.

Trains on 2018–2022 observations from data/processed/india_fire_weather_final.csv,
validates on 2023, and tests on held-out 2024–2025 data.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "india_fire_weather_final.csv"
OUT_DIR = BASE_DIR / "results" / "final_model"
MODEL_OUT = OUT_DIR / "final_hgb_model.joblib"
METRICS_OUT = OUT_DIR / "metrics.json"

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


def evaluate_split(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing validated dataset at {DATA_PATH}")

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    train_df = df[df["year"] <= 2022]
    val_df = df[df["year"] == 2023]
    test_df = df[df["year"] >= 2024]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train, y_train = train_df[FEATURES], train_df["fire"].to_numpy()
    X_val, y_val = val_df[FEATURES], val_df["fire"].to_numpy()
    X_test, y_test = test_df[FEATURES], test_df["fire"].to_numpy()

    print("Fitting HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_split(y_val, val_probs)
    test_metrics = evaluate_split(y_test, test_probs)

    test_preds = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, test_preds).tolist()

    summary = {
        "dataset": {
            "rows": len(df),
            "columns": len(df.columns),
            "date_min": str(df["acq_date"].min()),
            "date_max": str(df["acq_date"].max()),
            "fire_rows": int((df["fire"] == 1).sum()),
            "nonfire_rows": int((df["fire"] == 0).sum()),
        },
        "features": FEATURES,
        "model": {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "random_state": 42,
        },
        "split": {
            "train_years": "2018-2022",
            "validation_year": 2023,
            "test_years": "2024-2025",
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
        },
        "validation": val_metrics,
        "test": test_metrics,
        "test_confusion_matrix": cm,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(model, MODEL_OUT)
    print(f"Model and metrics saved successfully to {OUT_DIR}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
