"""Comprehensive baseline modeling suite for India wildfire forward forecasting.

Implements and benchmarks:
1. Logistic Regression (Linear baseline with standardized features)
2. Random Forest Classifier (Bagged ensemble)
3. HistGradientBoostingClassifier (Retained mini-project baseline)
4. LightGBM Classifier (Modern gradient boosting baseline)
5. Multi-Layer Perceptron (Neural baseline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.calibration import ModelCalibrator
from src.evaluation.metrics import compute_classification_metrics

# Standard 31 mini-project features
FEATURES_BASELINE_31 = [
    "grid_lat", "grid_lon", "hour", "year", "month",
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d",
    "temp_3d_mean", "temp_3d_max", "temp_3d_min", "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max", "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    "temp_7d_mean", "temp_7d_max", "temp_7d_min", "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max", "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]

# Complete 39 multimodal features
FEATURES_MULTIMODAL_39 = FEATURES_BASELINE_31 + [
    "elevation_m", "slope_deg", "ruggedness_index",
    "vpd_1d", "vpd_3d_mean", "soil_drought_index",
    "fire_history_recurrence", "antecedent_fire_24h",
]


def train_and_eval_model(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target_name: str = "fire",
) -> tuple[dict, np.ndarray, object]:
    """Train estimator, calibrate probabilities on validation set, and evaluate on test set."""
    print(f"Training {model_name} on {target_name} ({len(X_train):,} samples)...", flush=True)
    estimator.fit(X_train, y_train)

    # Validation predictions for calibration
    val_probs = estimator.predict_proba(X_val)[:, 1]
    calibrator = ModelCalibrator(method="isotonic")
    calibrator.fit(val_probs, y_val)

    # Test predictions
    raw_test_probs = estimator.predict_proba(X_test)[:, 1]
    cal_test_probs = calibrator.calibrate(raw_test_probs)

    uncal_metrics = compute_classification_metrics(y_test, raw_test_probs)
    cal_metrics = compute_classification_metrics(y_test, cal_test_probs)

    metrics = {
        "model": model_name,
        "target": target_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "uncalibrated": uncal_metrics,
        "calibrated": cal_metrics,
    }
    return metrics, cal_test_probs, estimator


def run_all_baselines(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    target_col: str = "fire",
    feature_set: str = "multimodal",
) -> pd.DataFrame:
    """Train and evaluate the complete baseline suite."""
    print(f"Loading datasets: Train={train_path}, Val={val_path}, Test={test_path}...", flush=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    features = FEATURES_MULTIMODAL_39 if feature_set == "multimodal" else FEATURES_BASELINE_31

    X_train, y_train = train_df[features], train_df[target_col].values
    X_val, y_val = val_df[features], val_df[target_col].values
    X_test, y_test = test_df[features], test_df[target_col].values

    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
        ),
        "MLP_Neural_Baseline": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=100,
                early_stopping=True,
                random_state=42,
            )),
        ]),
    }

    all_metrics = []
    test_predictions_df = pd.DataFrame({
        "grid_lat": test_df["grid_lat"],
        "grid_lon": test_df["grid_lon"],
        "acq_date": test_df["acq_date"],
        "y_true": y_test,
        "ecological_regime": test_df.get("ecological_regime", "UNKNOWN"),
    })

    for name, estimator in models.items():
        m, probs, fitted = train_and_eval_model(
            name, estimator, X_train, y_train, X_val, y_val, X_test, y_test, target_name=target_col
        )
        all_metrics.append(m)
        test_predictions_df[f"prob_{name}"] = np.round(probs, 4)
        
        # Save model artifact
        joblib.dump(fitted, output_dir / f"{name}.joblib")

    # Save summary table
    summary_rows = []
    for m in all_metrics:
        row = {"model": m["model"], "target": m["target"]}
        for k, v in m["calibrated"].items():
            row[f"cal_{k}"] = v
        for k, v in m["uncalibrated"].items():
            row[f"raw_{k}"] = v
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("cal_roc_auc", ascending=False)
    summary_df.to_csv(output_dir / "baseline_comparison_metrics.csv", index=False)
    test_predictions_df.to_csv(output_dir / "baseline_test_predictions.csv", index=False)
    
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n=== Baseline Comparison (Calibrated Test Metrics on 2024-2025) ===")
    print(summary_df[["model", "cal_accuracy", "cal_f1", "cal_roc_auc", "cal_pr_auc", "cal_brier_score", "cal_ece"]].to_string(index=False))
    return summary_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train_chronological.csv")
    p.add_argument("--val", default="data/splits/val_chronological.csv")
    p.add_argument("--test", default="data/splits/test_chronological.csv")
    p.add_argument("--output-dir", default="results/baselines")
    p.add_argument("--target", default="fire")
    p.add_argument("--feature-set", default="multimodal")
    args = p.parse_args()

    run_all_baselines(
        Path(args.train),
        Path(args.val),
        Path(args.test),
        Path(args.output_dir),
        target_col=args.target,
        feature_set=args.feature_set,
    )


if __name__ == "__main__":
    main()
