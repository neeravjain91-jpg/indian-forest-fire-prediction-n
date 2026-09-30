"""Evaluation of baseline classifiers and feature ablation groups.

Reproduces the core comparative experiments on the validated 131,000-row
dataset (data/processed/india_fire_weather_final.csv) using the strict
chronological split:
- Train: 2018–2022 (84,661 rows)
- Test: 2024–2025 (31,525 rows)

Compares:
1. Baseline Families: Logistic Regression, Random Forest, HistGradientBoosting
2. Feature Ablations:
   - Full 31 Features
   - Spatial Coordinates + Temporal (5 features)
   - Spatial Coordinates + Multi-timescale Weather (28 features)
   - Multi-timescale Weather Only (26 features)
   - Spatial Coordinates Only (2 features)
   - Temporal Only (3 features)
"""
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baselines_ablations")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "india_fire_weather_final.csv"
OUT_PATH = BASE_DIR / "results" / "baselines_and_ablations.json"

SPATIAL_FEATURES = ["grid_lat", "grid_lon"]
TEMPORAL_FEATURES = ["hour", "year", "month"]
WEATHER_1D = ["temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d"]
WEATHER_3D = [
    "temp_3d_mean", "temp_3d_max", "temp_3d_min",
    "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max",
    "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
]
WEATHER_7D = [
    "temp_7d_mean", "temp_7d_max", "temp_7d_min",
    "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max",
    "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]
ALL_WEATHER = WEATHER_1D + WEATHER_3D + WEATHER_7D
ALL_31_FEATURES = SPATIAL_FEATURES + TEMPORAL_FEATURES + ALL_WEATHER


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
    }


def run_experiments():
    logger.info(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    train_df = df[df["year"] <= 2022]
    test_df = df[df["year"] >= 2024]

    y_train = train_df["fire"].to_numpy()
    y_test = test_df["fire"].to_numpy()

    logger.info(f"Split sizes: Train={len(train_df)} rows, Test={len(test_df)} rows")

    results = {
        "dataset": {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "split": "Train 2018-2022, Test 2024-2025"
        },
        "baselines": {},
        "ablations": {}
    }

    # 1. BASELINES (All 31 features)
    X_train_31 = train_df[ALL_31_FEATURES]
    X_test_31 = test_df[ALL_31_FEATURES]

    logger.info("Evaluating Baseline 1: Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train_31, y_train)
    prob_lr = lr_pipe.predict_proba(X_test_31)[:, 1]
    results["baselines"]["logistic_regression"] = evaluate_predictions(y_test, prob_lr)

    logger.info("Evaluating Baseline 2: Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_31, y_train)
    prob_rf = rf.predict_proba(X_test_31)[:, 1]
    results["baselines"]["random_forest"] = evaluate_predictions(y_test, prob_rf)

    logger.info("Evaluating Primary Model: HistGradientBoosting (max_iter=300, lr=0.05, leaves=31)...")
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    hgb.fit(X_train_31, y_train)
    prob_hgb = hgb.predict_proba(X_test_31)[:, 1]
    results["baselines"]["hist_gradient_boosting"] = evaluate_predictions(y_test, prob_hgb)

    # 2. ABLATION EXPERIMENTS (using HGB architecture)
    ablation_groups = {
        "full_31_features": ALL_31_FEATURES,
        "coordinates_plus_temporal": SPATIAL_FEATURES + TEMPORAL_FEATURES,
        "coordinates_plus_weather": SPATIAL_FEATURES + ALL_WEATHER,
        "weather_only": ALL_WEATHER,
        "coordinates_only": SPATIAL_FEATURES,
        "temporal_only": TEMPORAL_FEATURES,
    }

    for name, feat_subset in ablation_groups.items():
        logger.info(f"Evaluating Ablation: {name} ({len(feat_subset)} features)...")
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        clf.fit(train_df[feat_subset], y_train)
        prob_ab = clf.predict_proba(test_df[feat_subset])[:, 1]
        metrics = evaluate_predictions(y_test, prob_ab)
        metrics["feature_count"] = len(feat_subset)
        metrics["features"] = feat_subset
        results["ablations"][name] = metrics

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results successfully saved to {OUT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_experiments()
