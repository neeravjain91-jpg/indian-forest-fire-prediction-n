"""Leave-One-Geographic-Regime-Out (LOGRO) Spatial Cross-Validation.

Stress-tests geographic generalizability across 6 predefined geographic fire regimes:
1. CENTRAL (Interior Deccan plateau core)
2. WESTERN_GHATS (Moist western coastal escarpment)
3. NORTHEAST (Purvanchal & Brahmaputra basin)
4. NORTH (Western Himalayas and Siwalik pine forests)
5. EAST (Eastern Ghats & Chota Nagpur plateau)
6. NORTHWEST (Semi-arid Thar & Aravalli scrub)

Methodological Note:
These regimes are predefined latitudinal-longitudinal macro-climatic partitions
rather than official WWF Terrestrial Ecoregions or WII biogeographic boundary polygons.
To prevent spatial autocorrelation leakage between the training partition and calibration split,
temporal validation (Train <= 2022, Val = 2023) is enforced within the training regimes,
leaving the held-out geographic regime completely untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from src.evaluation.calibration import ModelCalibrator
from src.evaluation.metrics import compute_classification_metrics
from src.models.baselines import FEATURES_BASELINE_31, FEATURES_MULTIMODAL_39

REGIMES = ["CENTRAL", "WESTERN_GHATS", "NORTHEAST", "NORTH", "EAST", "NORTHWEST"]


def run_loeo_generalization(
    splits_dir: Path,
    output_dir: Path,
    target_col: str = "fire",
) -> pd.DataFrame:
    """Run full Leave-One-Geographic-Regime-Out cross-validation across all 6 regimes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for held_out in REGIMES:
        train_file = splits_dir / f"train_loeo_exclude_{held_out.lower()}.csv"
        test_file = splits_dir / f"test_loeo_holdout_{held_out.lower()}.csv"

        if not train_file.exists() or not test_file.exists():
            print(f"Skipping {held_out}: split files not found.", flush=True)
            continue

        print(f"\n--- LOGRO Fold: Evaluating Held-Out Regime {held_out} ---", flush=True)
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        y_test = test_df[target_col].values

        # Temporal validation within training regions (prevents spatial autocorrelation leakage)
        # tr_sub: historical training (<= 2022); va_sub: calibration split (2023)
        if "year" in train_df.columns:
            tr_sub = train_df[train_df["year"] <= 2022]
            va_sub = train_df[train_df["year"] == 2023]
        else:
            tr_sub, va_sub = train_df, train_df.head(1000)

        if len(va_sub) < 500 or len(tr_sub) < 1000:
            val_mask = np.random.default_rng(42).random(len(train_df)) < 0.15
            tr_sub = train_df[~val_mask]
            va_sub = train_df[val_mask]

        models = [
            ("HGB_31_Baseline", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=42), FEATURES_BASELINE_31),
            ("LGBM_39_Multimodal", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1), FEATURES_MULTIMODAL_39),
        ]

        for mod_name, clf, feats in models:
            clf.fit(tr_sub[feats], tr_sub[target_col].values)
            val_p = clf.predict_proba(va_sub[feats])[:, 1]
            calibrator_platt = ModelCalibrator(method="platt").fit(val_p, va_sub[target_col].values)
            calibrator_iso = ModelCalibrator(method="isotonic").fit(val_p, va_sub[target_col].values)

            raw_p = clf.predict_proba(test_df[feats])[:, 1]
            p_platt = calibrator_platt.calibrate(raw_p)
            p_iso = calibrator_iso.calibrate(raw_p)

            m_raw = compute_classification_metrics(y_test, raw_p)
            m_platt = compute_classification_metrics(y_test, p_platt)
            m_iso = compute_classification_metrics(y_test, p_iso)

            records.append({
                "held_out_region": held_out,
                "model_id": mod_name,
                "feature_count": len(feats),
                "n_train": len(tr_sub),
                "n_val": len(va_sub),
                "n_test": len(test_df),
                "test_fire_rate": round(float(np.mean(y_test)), 4),
                "accuracy": m_platt["accuracy"],
                "f1": m_platt["f1"],
                "roc_auc": m_platt["roc_auc"],
                "pr_auc": m_platt["pr_auc"],
                "brier_score": m_platt["brier_score"],
                "ece": m_platt["ece"],
                "mce": m_platt["mce"],
                "raw_brier": m_raw["brier_score"],
                "raw_ece": m_raw["ece"],
                "iso_brier": m_iso["brier_score"],
                "iso_ece": m_iso["ece"],
            })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(output_dir / "loeo_geographic_metrics.csv", index=False)

    # Compute Aggregate Mean ± Std across regions
    agg_df = summary_df.groupby("model_id").agg(
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        mean_pr_auc=("pr_auc", "mean"),
        std_pr_auc=("pr_auc", "std"),
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_brier=("brier_score", "mean"),
        mean_ece=("ece", "mean"),
    ).reset_index()

    for col in ["mean_roc_auc", "std_roc_auc", "mean_pr_auc", "std_pr_auc", "mean_f1", "std_f1", "mean_brier", "mean_ece"]:
        agg_df[col] = agg_df[col].round(4)

    agg_df.to_csv(output_dir / "loeo_aggregate_summary.csv", index=False)

    print("\n=== Leave-One-Ecoregion-Out (LOEO) Geographic Results Summary ===")
    print(summary_df[["held_out_region", "model_id", "accuracy", "f1", "roc_auc", "pr_auc", "brier_score", "ece"]].to_string(index=False))

    print("\n=== Cross-Regional Macro Mean ± Standard Deviation ===")
    print(agg_df.to_string(index=False))

    return summary_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--output-dir", default="results/geographic")
    args = p.parse_args()

    run_loeo_generalization(Path(args.splits_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
