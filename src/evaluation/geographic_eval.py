"""Geographic & Ecological Generalization Evaluation.

Stress-tests models across spatially disjoint ecoregions:
Trains on non-Central India biomes (83k+ samples) and evaluates on the
held-out Central Deciduous / Deccan Plateau biome (47k+ samples).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.calibration import ModelCalibrator
from src.evaluation.metrics import compute_classification_metrics
from src.models.baselines import FEATURES_BASELINE_31, FEATURES_MULTIMODAL_39


def run_geographic_generalization(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    target_col: str = "fire",
) -> pd.DataFrame:
    """Run spatially disjoint evaluation and measure transfer degradation."""
    print(f"Loading spatial splits: Train={train_path}, Test={test_path}...", flush=True)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Use 80/20 train/val split inside the training region for calibration
    val_sample_mask = np.random.default_rng(42).random(len(train_df)) < 0.15
    train_part = train_df[~val_sample_mask]
    val_part = train_df[val_sample_mask]

    output_dir.mkdir(parents=True, exist_ok=True)

    models_config = [
        ("Logistic_Regression_Baseline", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]), FEATURES_BASELINE_31),
        ("HGB_Mini_Baseline", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=42), FEATURES_BASELINE_31),
        ("LightGBM_Multimodal", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1), FEATURES_MULTIMODAL_39),
        ("Random_Forest_Multimodal", RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1), FEATURES_MULTIMODAL_39),
    ]

    records = []

    for name, estimator, features in models_config:
        print(f"Running geographic holdout test for {name} ({len(features)} features)...", flush=True)
        X_tr = train_part[features]
        y_tr = train_part[target_col].values
        X_va = val_part[features]
        y_va = val_part[target_col].values
        X_te = test_df[features]
        y_te = test_df[target_col].values

        estimator.fit(X_tr, y_tr)

        # Calibrate on validation split from training ecoregions
        val_probs = estimator.predict_proba(X_va)[:, 1]
        calibrator = ModelCalibrator(method="isotonic").fit(val_probs, y_va)

        raw_test_probs = estimator.predict_proba(X_te)[:, 1]
        cal_test_probs = calibrator.calibrate(raw_test_probs)

        uncal_m = compute_classification_metrics(y_te, raw_test_probs)
        cal_m = compute_classification_metrics(y_te, cal_test_probs)

        records.append({
            "model": name,
            "feature_set": "31_baseline" if len(features) == 31 else "39_multimodal",
            "train_regimes": "NON_CENTRAL (North, NE, West, East, NW)",
            "test_regime": "CENTRAL (Deccan Dry Deciduous)",
            "n_train": len(train_part),
            "n_test": len(test_df),
            "accuracy": cal_m["accuracy"],
            "f1": cal_m["f1"],
            "roc_auc": cal_m["roc_auc"],
            "pr_auc": cal_m["pr_auc"],
            "brier_score": cal_m["brier_score"],
            "ece": cal_m["ece"],
            "uncal_roc_auc": uncal_m["roc_auc"],
            "uncal_ece": uncal_m["ece"],
        })

    summary_df = pd.DataFrame(records).sort_values("roc_auc", ascending=False)
    summary_df.to_csv(output_dir / "geographic_holdout_metrics.csv", index=False)
    
    with open(output_dir / "geographic_holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n=== Spatially Disjoint Geographic Holdout Results (Central India Test) ===")
    print(summary_df[["model", "feature_set", "accuracy", "f1", "roc_auc", "pr_auc", "brier_score", "ece"]].to_string(index=False))
    return summary_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train_spatial_non_central.csv")
    p.add_argument("--test", default="data/splits/test_spatial_central.csv")
    p.add_argument("--output-dir", default="results/geographic")
    args = p.parse_args()

    run_geographic_generalization(
        Path(args.train),
        Path(args.test),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
