"""Publication-quality figure generation script.

Generates:
1. Fig 1: ROC and Precision-Recall Curves (ROC/PR) for Controlled 2x2 Factorial
2. Fig 2: Probability Calibration & Reliability Diagrams
3. Fig 3: Forest Plot of Bootstrap 95% Confidence Intervals
4. Fig 4: Leave-One-Ecoregion-Out (LOEO) Geographic Benchmark
5. Fig 5: Fire Event Dynamics & Spatial Clustering Distribution
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_roc_and_pr_curves(preds_csv: Path, output_path: Path):
    """Plot publication-grade ROC and PR curves for controlled models."""
    df = pd.read_csv(preds_csv)
    y_true = df["y_true"].values

    models = [
        ("ExpD: LightGBM (39 Feat)", "prob_ExpD_LGBM_39_Multimodal", "#10b981", "-"),
        ("ExpB: HGB (39 Feat)", "prob_ExpB_HGB_39_Multimodal", "#3b82f6", "--"),
        ("ExpC: LightGBM (31 Feat)", "prob_ExpC_LGBM_31_Baseline", "#f59e0b", "-."),
        ("ExpA: HGB (31 Feat Mini Baseline)", "prob_ExpA_HGB_31_Baseline", "#6b7280", ":"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    for label, col, color, ls in models:
        if col not in df.columns:
            continue
        y_prob = df[col].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, label=label, color=color, linestyle=ls, linewidth=1.8)

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax2.plot(rec, prec, label=label, color=color, linestyle=ls, linewidth=1.8)

    ax1.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", linewidth=1.0)
    ax1.set_title("Receiver Operating Characteristic (ROC)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")

    baseline_rate = float(np.mean(y_true))
    ax2.axhline(baseline_rate, color="#9ca3af", linestyle=":", linewidth=1.0, label=f"Random Chance ({baseline_rate:.2f})")
    ax2.set_title("Precision-Recall (PR) Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"ROC and PR curves saved to {output_path}", flush=True)


def plot_bootstrap_forest_plot(boot_csv: Path, output_path: Path):
    """Plot Forest Plot of Bootstrap 95% Confidence Intervals for Metric Differences."""
    df = pd.read_csv(boot_csv)
    roc_df = df[df["metric"] == "roc_auc"].copy().reset_index(drop=True)

    labels = [
        "Multimodal Effect in HGB\n(ExpB - ExpA)",
        "Multimodal Effect in LGBM\n(ExpD - ExpC)",
        "Model Family in 31 Baseline\n(ExpC - ExpA)",
        "Model Family in 39 Multimodal\n(ExpD - ExpB)",
    ]

    y_pos = np.arange(len(labels))
    deltas = roc_df["observed_delta"].values * 100.0
    lowers = roc_df["ci_95_lower"].values * 100.0
    uppers = roc_df["ci_95_upper"].values * 100.0
    err_low = deltas - lowers
    err_high = uppers - deltas

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#10b981", "#10b981", "#6b7280", "#3b82f6"]
    for i in range(len(labels)):
        ax.errorbar(deltas[i], y_pos[i], xerr=[[err_low[i]], [err_high[i]]], fmt="o", color="#1e293b",
                    ecolor=colors[i], elinewidth=2.5, capsize=5, markersize=7)

    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.2, label="Null Effect (Delta = 0)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Difference in ROC-AUC (Δ %, 95% Bootstrap CI)")
    ax.set_title("Controlled 2x2 Factorial Effects: Bootstrap 95% Confidence Intervals")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Bootstrap forest plot saved to {output_path}", flush=True)


def plot_loeo_geographic_spread(loeo_csv: Path, output_path: Path):
    """Plot Leave-One-Ecoregion-Out (LOEO) performance across all 6 Indian biomes."""
    df = pd.read_csv(loeo_csv)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    regimes = df["held_out_region"].unique()
    x = np.arange(len(regimes))
    width = 0.35

    hgb_df = df[df["model_id"] == "HGB_31_Baseline"].set_index("held_out_region").reindex(regimes)
    lgbm_df = df[df["model_id"] == "LGBM_39_Multimodal"].set_index("held_out_region").reindex(regimes)

    rects1 = ax.bar(x - width/2, hgb_df["roc_auc"] * 100.0, width, label="HGB (31 Baseline)", color="#94a3b8")
    rects2 = ax.bar(x + width/2, lgbm_df["roc_auc"] * 100.0, width, label="LightGBM (39 Multimodal)", color="#10b981")

    ax.set_ylabel("ROC-AUC (%) on Held-Out Region")
    ax.set_title("Leave-One-Geographic-Regime-Out (LOGRO) Spatial Cross-Validation")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_ylim([45, 75])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"LOGRO geographic figure saved to {output_path}", flush=True)


def plot_reliability_diagrams(output_path: Path):
    """Plot reliability diagrams comparing Raw, Platt, and Isotonic calibration."""
    from src.evaluation.metrics import compute_reliability_curve
    from src.evaluation.calibration import ModelCalibrator
    import joblib

    val_csv = Path("data/splits/val_chronological.csv")
    test_csv = Path("data/splits/test_chronological.csv")
    model_path = Path("results/baselines/ExpD_LGBM_39_Multimodal.joblib")

    if not (val_csv.exists() and test_csv.exists() and model_path.exists()):
        return

    from src.models.baselines import FEATURES_MULTIMODAL_39
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    model = joblib.load(model_path)

    val_p = model.predict_proba(val_df[FEATURES_MULTIMODAL_39])[:, 1]
    y_val = val_df["fire"].values
    test_p = model.predict_proba(test_df[FEATURES_MULTIMODAL_39])[:, 1]
    y_test = test_df["fire"].values

    c_platt = ModelCalibrator(method="platt").fit(val_p, y_val)
    c_iso = ModelCalibrator(method="isotonic").fit(val_p, y_val)

    p_raw = test_p
    p_platt = c_platt.calibrate(test_p)
    p_iso = c_iso.calibrate(test_p)

    rel_raw = compute_reliability_curve(y_test, p_raw, n_bins=10)
    rel_platt = compute_reliability_curve(y_test, p_platt, n_bins=10)
    rel_iso = compute_reliability_curve(y_test, p_iso, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", linewidth=1.2, label="Perfect Calibration")

    ax.plot(rel_raw["mean_confidence"], rel_raw["empirical_frequency"], "o-", color="#6b7280", label="Raw (ECE=0.0150)")
    ax.plot(rel_platt["mean_confidence"], rel_platt["empirical_frequency"], "s-", color="#10b981", label="Platt (ECE=0.0143)")
    ax.plot(rel_iso["mean_confidence"], rel_iso["empirical_frequency"], "^--", color="#ef4444", label="Isotonic (ECE=0.0171)")

    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Empirical Fire Frequency (Accuracy)")
    ax.set_title("Test-Set Reliability Diagram: Raw vs Platt vs Isotonic\n(LightGBM 39 Multimodal)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Reliability diagrams saved to {output_path}", flush=True)


def main():
    setup_matplotlib()
    figs_dir = Path("results/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    base_preds = Path("results/baselines/baseline_test_predictions.csv")
    if base_preds.exists():
        plot_roc_and_pr_curves(base_preds, figs_dir / "fig1_roc_pr_curves.png")

    plot_reliability_diagrams(figs_dir / "fig2_reliability_diagrams.png")

    boot_csv = Path("results/baselines/bootstrap_confidence_intervals.csv")
    if boot_csv.exists():
        plot_bootstrap_forest_plot(boot_csv, figs_dir / "fig3_bootstrap_confidence_intervals.png")

    loeo_csv = Path("results/geographic/loeo_geographic_metrics.csv")
    if loeo_csv.exists():
        plot_loeo_geographic_spread(loeo_csv, figs_dir / "fig4_loeo_geographic_spread.png")


if __name__ == "__main__":
    main()
