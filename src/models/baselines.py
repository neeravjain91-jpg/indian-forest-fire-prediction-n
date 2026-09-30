"""Controlled Baseline Modeling & 2x2 Multimodal Factorial Benchmark.

Implements:
1. Controlled 2x2 Factorial Experiment:
   - Exp A: HistGradientBoosting + 31 baseline features
   - Exp B: HistGradientBoosting + 39 multimodal features
   - Exp C: LightGBM + 31 baseline features
   - Exp D: LightGBM + 39 multimodal features
2. Additional Baselines: Logistic Regression, Random Forest, MLP
3. Non-parametric Bootstrap 95% Confidence Intervals for metric differences
4. Precision@k and Recall@k for class-imbalanced targets
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
from src.evaluation.statistical_testing import (
    compute_bootstrap_confidence_interval,
    compute_precision_recall_at_k,
)

# Standard 31 mini-project features
FEATURES_BASELINE_31 = [
    "grid_lat", "grid_lon", "hour", "year", "month",
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d",
    "temp_3d_mean", "temp_3d_max", "temp_3d_min", "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max", "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    "temp_7d_mean", "temp_7d_max", "temp_7d_min", "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max", "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]

# Complete 39 multimodal features (includes real DEM terrain, fuel dryness, causal recurrence)
FEATURES_MULTIMODAL_39 = FEATURES_BASELINE_31 + [
    "elevation_m", "slope_deg", "ruggedness_index",
    "vpd_1d", "vpd_3d_mean", "soil_drought_index",
    "fire_history_recurrence", "antecedent_fire_24h",
]


def train_eval_single(
    name: str,
    estimator,
    features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "fire",
) -> tuple[dict, dict[str, np.ndarray], object]:
    """Train model, fit Platt and Isotonic calibrators on validation set, evaluate on test set."""
    X_tr = train_df[features]
    y_tr = train_df[target_col].values
    X_va = val_df[features]
    y_va = val_df[target_col].values
    X_te = test_df[features]
    y_te = test_df[target_col].values

    estimator.fit(X_tr, y_tr)
    val_p = estimator.predict_proba(X_va)[:, 1]

    cal_platt = ModelCalibrator(method="platt").fit(val_p, y_va)
    cal_iso = ModelCalibrator(method="isotonic").fit(val_p, y_va)

    raw_test_p = estimator.predict_proba(X_te)[:, 1]
    platt_test_p = cal_platt.calibrate(raw_test_p)
    iso_test_p = cal_iso.calibrate(raw_test_p)

    raw_m = compute_classification_metrics(y_te, raw_test_p)
    platt_m = compute_classification_metrics(y_te, platt_test_p)
    iso_m = compute_classification_metrics(y_te, iso_test_p)

    probs = {
        "raw": raw_test_p,
        "platt": platt_test_p,
        "isotonic": iso_test_p,
    }

    return {
        "model_id": name,
        "feature_count": len(features),
        "target": target_col,
        "raw": raw_m,
        "platt": platt_m,
        "isotonic": iso_m,
        # Default cal maps to platt which regularizes without inflating test ECE
        "calibrated": platt_m,
        "uncalibrated": raw_m,
    }, probs, estimator


def run_controlled_baseline_experiments(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    target_col: str = "fire",
) -> pd.DataFrame:
    """Run controlled 2x2 factorial experiment, calibration comparison, and benchmark suite."""
    print("Loading datasets for controlled baseline benchmarks...", flush=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col].values

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Controlled 2x2 Factorial Configurations
    controlled_configs = [
        ("ExpA_HGB_31_Baseline", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=42), FEATURES_BASELINE_31),
        ("ExpB_HGB_39_Multimodal", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=42), FEATURES_MULTIMODAL_39),
        ("ExpC_LGBM_31_Baseline", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1), FEATURES_BASELINE_31),
        ("ExpD_LGBM_39_Multimodal", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1), FEATURES_MULTIMODAL_39),
        ("Logistic_Regression_39", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]), FEATURES_MULTIMODAL_39),
        ("Random_Forest_39", RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1), FEATURES_MULTIMODAL_39),
    ]

    all_results = []
    test_preds_df = pd.DataFrame({
        "grid_lat": test_df["grid_lat"],
        "grid_lon": test_df["grid_lon"],
        "acq_date": test_df["acq_date"],
        "y_true": y_test,
        "geographic_regime": test_df.get("ecological_regime", "UNKNOWN"),
    })
    stored_probs = {}

    for name, clf, feats in controlled_configs:
        print(f"Training {name} ({len(feats)} features)...", flush=True)
        res, probs_dict, fitted = train_eval_single(name, clf, feats, train_df, val_df, test_df, target_col=target_col)
        all_results.append(res)
        test_preds_df[f"prob_{name}"] = np.round(probs_dict["platt"], 4)
        stored_probs[name] = probs_dict
        joblib.dump(fitted, output_dir / f"{name}.joblib")

    # Save summary table
    summary_rows = []
    cal_comp_rows = []
    for r in all_results:
        row = {"model_id": r["model_id"], "features": r["feature_count"]}
        for k, v in r["platt"].items():
            row[f"cal_{k}"] = v
            row[f"platt_{k}"] = v
        for k, v in r["raw"].items():
            row[f"raw_{k}"] = v
        for k, v in r["isotonic"].items():
            row[f"iso_{k}"] = v
        summary_rows.append(row)

        for m_name in ["raw", "platt", "isotonic"]:
            cal_comp_rows.append({
                "model_id": r["model_id"],
                "features": r["feature_count"],
                "calibration_method": m_name,
                **r[m_name],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "baseline_comparison_metrics.csv", index=False)
    test_preds_df.to_csv(output_dir / "baseline_test_predictions.csv", index=False)

    cal_comp_df = pd.DataFrame(cal_comp_rows)
    cal_comp_df.to_csv(output_dir / "calibration_comparison.csv", index=False)

    # 2. Compute True 2x2 Factorial Interaction with Paired Bootstrap
    print("\nComputing True 2x2 Factorial Interaction & 95% Bootstrap CIs (B=1000)...", flush=True)
    from src.evaluation.statistical_testing import compute_factorial_interaction_bootstrap

    fact_records = []
    prob_A = stored_probs["ExpA_HGB_31_Baseline"]["platt"]
    prob_B = stored_probs["ExpB_HGB_39_Multimodal"]["platt"]
    prob_C = stored_probs["ExpC_LGBM_31_Baseline"]["platt"]
    prob_D = stored_probs["ExpD_LGBM_39_Multimodal"]["platt"]

    for metric in ["roc_auc", "pr_auc", "brier", "f1"]:
        fact_dict = compute_factorial_interaction_bootstrap(y_test, prob_A, prob_B, prob_C, prob_D, metric_name=metric, n_bootstraps=1000)
        for eff_name, eff_vals in fact_dict.items():
            fact_records.append(eff_vals)

    fact_df = pd.DataFrame(fact_records)
    fact_df.to_csv(output_dir / "factorial_interaction_analysis.csv", index=False)

    # Pairwise Bootstrap Confidence Intervals (preserving baseline format)
    comparisons = [
        ("Multimodal_Effect_in_HGB", "ExpB_HGB_39_Multimodal", "ExpA_HGB_31_Baseline"),
        ("Multimodal_Effect_in_LGBM", "ExpD_LGBM_39_Multimodal", "ExpC_LGBM_31_Baseline"),
        ("Model_Family_Effect_in_31_Baseline", "ExpC_LGBM_31_Baseline", "ExpA_HGB_31_Baseline"),
        ("Model_Family_Effect_in_39_Multimodal", "ExpD_LGBM_39_Multimodal", "ExpB_HGB_39_Multimodal"),
    ]

    boot_records = []
    for label, mod_a, mod_b in comparisons:
        p_a = stored_probs[mod_a]["platt"]
        p_b = stored_probs[mod_b]["platt"]
        for metric in ["roc_auc", "pr_auc", "brier", "f1"]:
            ci_res = compute_bootstrap_confidence_interval(y_test, p_a, p_b, metric_name=metric, n_bootstraps=1000)
            boot_records.append({
                "comparison": label,
                "model_a": mod_a,
                "model_b": mod_b,
                **ci_res,
            })

    boot_df = pd.DataFrame(boot_records)
    boot_df.to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)

    # 3. Precision@k and Recall@k for Rare-Event Targets
    print("Computing Precision@k and Recall@k...", flush=True)
    pk_records = []
    for name in ["ExpA_HGB_31_Baseline", "ExpB_HGB_39_Multimodal", "ExpD_LGBM_39_Multimodal"]:
        pk_df = compute_precision_recall_at_k(y_test, stored_probs[name]["platt"], k_list=[100, 250, 500, 1000, 2500])
        pk_df["model_id"] = name
        pk_records.append(pk_df)
    pd.concat(pk_records, ignore_index=True).to_csv(output_dir / "precision_recall_at_k.csv", index=False)

    print("\n=== Calibration Comparison (Raw vs Platt vs Isotonic) ===")
    print(cal_comp_df[["model_id", "calibration_method", "brier_score", "ece", "mce", "roc_auc", "f1"]].to_string(index=False))

    print("\n=== True 2x2 Factorial Interaction Analysis (95% Bootstrap CIs) ===")
    print(fact_df[["effect_name", "metric", "observed", "ci_95_lower", "ci_95_upper", "ci_excludes_zero"]].to_string(index=False))

    return summary_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train_chronological.csv")
    p.add_argument("--val", default="data/splits/val_chronological.csv")
    p.add_argument("--test", default="data/splits/test_chronological.csv")
    p.add_argument("--output-dir", default="results/baselines")
    p.add_argument("--target", default="fire")
    args = p.parse_args()

    run_controlled_baseline_experiments(
        Path(args.train),
        Path(args.val),
        Path(args.test),
        Path(args.output_dir),
        target_col=args.target,
    )


if __name__ == "__main__":
    main()
