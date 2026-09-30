"""Ablation benchmark quantifying the marginal contribution of each modality and calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.evaluation.calibration import ModelCalibrator
from src.evaluation.metrics import compute_classification_metrics

# Modality Feature Subsets
FEAT_WEATHER_1D = [
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d"
]

FEAT_WEATHER_MULTI = FEAT_WEATHER_1D + [
    "temp_3d_mean", "temp_3d_max", "temp_3d_min", "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max", "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    "temp_7d_mean", "temp_7d_max", "temp_7d_min", "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max", "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]

FEAT_FIRE_HISTORY = [
    "fire_history_recurrence", "antecedent_fire_24h"
]

FEAT_TERRAIN_ENV = [
    "elevation_m", "slope_deg", "ruggedness_index",
    "vpd_1d", "vpd_3d_mean", "soil_drought_index"
]

FEAT_COORDINATES = [
    "grid_lat", "grid_lon", "hour", "year", "month"
]


def run_ablation_experiments(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    target_col: str = "fire",
) -> pd.DataFrame:
    """Run controlled ablation study across identical splits."""
    print("Loading datasets for ablation study...", flush=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    output_dir.mkdir(parents=True, exist_ok=True)

    ablations = [
        ("A_Weather_1d_Only", FEAT_WEATHER_1D, "Instantaneous 1d weather alone"),
        ("B_Weather_MultiTimescale", FEAT_WEATHER_MULTI, "Multi-timescale (1d + 3d + 7d) atmospheric drying"),
        ("C_Weather_Plus_FireHistory", FEAT_WEATHER_MULTI + FEAT_FIRE_HISTORY, "Weather history + Antecedent persistence"),
        ("D_Weather_Plus_Terrain_Env", FEAT_WEATHER_MULTI + FEAT_TERRAIN_ENV, "Weather history + Terrain geomorphology & VPD"),
        ("E_Full_Multimodal", FEAT_WEATHER_MULTI + FEAT_TERRAIN_ENV + FEAT_FIRE_HISTORY + FEAT_COORDINATES, "Full multimodal integration (39 features)"),
    ]

    records = []

    for name, feat_subset, desc in ablations:
        print(f"\nEvaluating Ablation: {name} ({len(feat_subset)} features)...", flush=True)
        clf = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(train_df[feat_subset], y_train)

        val_p = clf.predict_proba(val_df[feat_subset])[:, 1]
        calibrator = ModelCalibrator(method="isotonic").fit(val_p, y_val)

        raw_test_p = clf.predict_proba(test_df[feat_subset])[:, 1]
        cal_test_p = calibrator.calibrate(raw_test_p)

        uncal_m = compute_classification_metrics(y_test, raw_test_p)
        cal_m = compute_classification_metrics(y_test, cal_test_p)

        records.append({
            "ablation_id": name,
            "description": desc,
            "feature_count": len(feat_subset),
            "cal_accuracy": cal_m["accuracy"],
            "cal_f1": cal_m["f1"],
            "cal_roc_auc": cal_m["roc_auc"],
            "cal_pr_auc": cal_m["pr_auc"],
            "cal_brier_score": cal_m["brier_score"],
            "cal_ece": cal_m["ece"],
            "uncal_roc_auc": uncal_m["roc_auc"],
            "uncal_ece": uncal_m["ece"],
        })

    # Add uncalibrated full model row for comparison
    records.append({
        "ablation_id": "F_Full_Multimodal_Uncalibrated",
        "description": "Full multimodal model without post-hoc probability calibration",
        "feature_count": len(ablations[-1][1]),
        "cal_accuracy": uncal_m["accuracy"],
        "cal_f1": uncal_m["f1"],
        "cal_roc_auc": uncal_m["roc_auc"],
        "cal_pr_auc": uncal_m["pr_auc"],
        "cal_brier_score": uncal_m["brier_score"],
        "cal_ece": uncal_m["ece"],
        "uncal_roc_auc": uncal_m["roc_auc"],
        "uncal_ece": uncal_m["ece"],
    })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(output_dir / "ablation_comparison.csv", index=False)
    
    with open(output_dir / "ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n=== Controlled Modality Ablation Matrix (Test Set 2024-2025) ===")
    print(summary_df[["ablation_id", "feature_count", "cal_accuracy", "cal_f1", "cal_roc_auc", "cal_pr_auc", "cal_brier_score", "cal_ece"]].to_string(index=False))
    return summary_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train_chronological.csv")
    p.add_argument("--val", default="data/splits/val_chronological.csv")
    p.add_argument("--test", default="data/splits/test_chronological.csv")
    p.add_argument("--output-dir", default="results/ablations")
    args = p.parse_args()

    run_ablation_experiments(
        Path(args.train),
        Path(args.val),
        Path(args.test),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
