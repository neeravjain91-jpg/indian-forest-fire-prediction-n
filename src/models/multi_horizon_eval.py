"""Multi-Horizon Forecasting and Event Evolution Evaluation.

Evaluates predictive performance across defensible forecast horizons:
1. T (Synchronous baseline)
2. T+24h (Next-Day Forward Forecast)
3. T+48h (Two-Day Forward Forecast)
4. Event Persistence (Local active complex continuation into T+24h)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.evaluation.calibration import ModelCalibrator
from src.evaluation.metrics import compute_classification_metrics
from src.models.baselines import FEATURES_MULTIMODAL_39


def evaluate_multi_horizon(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Train and evaluate forward-looking multi-horizon models."""
    print(f"Loading datasets for multi-horizon evaluation...", flush=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = [
        ("Occurrence_T", "fire", "Instantaneous detection reference at time T"),
        ("Forward_T_plus_24h", "target_fire_lead_24h", "Next-day 24h forward forecast"),
        ("Forward_T_plus_48h", "target_fire_lead_48h", "Two-day 48h forward forecast"),
        ("Event_Persistence_24h", "target_event_persistence", "Active complex continuation into T+24h"),
    ]

    records = []

    for name, target_col, desc in horizons:
        if target_col not in train_df.columns:
            print(f"Skipping {name}: target column {target_col} not found.", flush=True)
            continue

        print(f"\n--- Training Horizon: {name} (Target: {target_col}) ---", flush=True)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        pos_rate_tr = np.mean(y_train)
        pos_rate_te = np.mean(y_test)
        print(f"Train positive rate: {pos_rate_tr:.2%}, Test positive rate: {pos_rate_te:.2%}", flush=True)

        clf = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(train_df[FEATURES_MULTIMODAL_39], y_train)

        val_probs = clf.predict_proba(val_df[FEATURES_MULTIMODAL_39])[:, 1]
        calibrator = ModelCalibrator(method="isotonic").fit(val_probs, y_val)

        raw_probs = clf.predict_proba(test_df[FEATURES_MULTIMODAL_39])[:, 1]
        cal_probs = calibrator.calibrate(raw_probs)

        uncal_m = compute_classification_metrics(y_test, raw_probs)
        cal_m = compute_classification_metrics(y_test, cal_probs)

        records.append({
            "horizon": name,
            "target_variable": target_col,
            "description": desc,
            "positive_rate_test": round(float(pos_rate_te), 4),
            "cal_accuracy": cal_m["accuracy"],
            "cal_f1": cal_m["f1"],
            "cal_roc_auc": cal_m["roc_auc"],
            "cal_pr_auc": cal_m["pr_auc"],
            "cal_brier_score": cal_m["brier_score"],
            "cal_ece": cal_m["ece"],
            "raw_roc_auc": uncal_m["roc_auc"],
            "raw_ece": uncal_m["ece"],
        })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(output_dir / "multi_horizon_comparison.csv", index=False)
    
    with open(output_dir / "multi_horizon_metrics.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n=== Multi-Horizon Forecasting Evaluation Summary ===")
    print(summary_df[["horizon", "positive_rate_test", "cal_accuracy", "cal_f1", "cal_roc_auc", "cal_pr_auc", "cal_brier_score", "cal_ece"]].to_string(index=False))
    return summary_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train_chronological.csv")
    p.add_argument("--val", default="data/splits/val_chronological.csv")
    p.add_argument("--test", default="data/splits/test_chronological.csv")
    p.add_argument("--output-dir", default="results/multi_horizon")
    args = p.parse_args()

    evaluate_multi_horizon(
        Path(args.train),
        Path(args.val),
        Path(args.test),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
