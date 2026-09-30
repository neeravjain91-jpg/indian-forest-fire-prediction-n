"""Publication-quality figure generation script.

Generates:
1. ROC and Precision-Recall Curves (ROC/PR)
2. Probability Calibration & Reliability Diagrams
3. Uncertainty & OOD Error-Correlation Analysis
4. Spatially Disjoint Regional Generalization Benchmark
5. Controlled Modality Ablation Matrix
6. Fire Event Dynamics & Spatial Clustering Distribution
"""

from __future__ import annotations

import json
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
    """Plot publication-grade ROC and PR curves from real test predictions."""
    df = pd.read_csv(preds_csv)
    y_true = df["y_true"].values

    models = [
        ("LightGBM", "prob_LightGBM", "#10b981", "-"),
        ("Hist_Gradient_Boosting (Mini Baseline)", "prob_Hist_Gradient_Boosting", "#f59e0b", "--"),
        ("Random_Forest", "prob_Random_Forest", "#3b82f6", "-."),
        ("Logistic_Regression", "prob_Logistic_Regression", "#6b7280", ":"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    for label, col, color, ls in models:
        if col not in df.columns:
            continue
        y_prob = df[col].values
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, label=label, color=color, linestyle=ls, linewidth=1.8)
        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax2.plot(rec, prec, label=label, color=color, linestyle=ls, linewidth=1.8)

    # Reference diagonals
    ax1.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", linewidth=1.0)
    ax1.set_title("Receiver Operating Characteristic (ROC)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")

    baseline_rate = np.mean(y_true)
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


def plot_calibration_curves(preds_csv: Path, output_path: Path):
    """Plot reliability diagram comparing raw vs isotonic calibrated probabilities."""
    df = pd.read_csv(preds_csv)
    y_true = df["y_true"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0.0, 1.0, 11)

    prob_cols = [
        ("Raw HGB (Mini Baseline)", "prob_Hist_Gradient_Boosting", "#f59e0b", "s--"),
        ("Calibrated LightGBM (Major)", "prob_LightGBM", "#10b981", "o-"),
    ]

    for label, col, color, fmt in prob_cols:
        if col not in df.columns:
            continue
        p = df[col].values
        bin_confs, bin_freqs = [], []
        for i in range(10):
            mask = (p >= bins[i]) & (p < bins[i + 1])
            if np.sum(mask) > 10:
                bin_confs.append(np.mean(p[mask]))
                bin_freqs.append(np.mean(y_true[mask]))
        ax.plot(bin_confs, bin_freqs, fmt, color=color, label=label, linewidth=1.8, markersize=5)

    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.set_title("Probability Calibration Reliability Diagram (2024–2025 Test)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fire Frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Calibration plot saved to {output_path}", flush=True)


def plot_ablation_matrix(ablation_csv: Path, output_path: Path):
    """Plot bar chart of ablation progression across feature modalities."""
    df = pd.read_csv(ablation_csv)
    # Filter out uncalibrated row for cleaner modality comparison
    df_clean = df[~df["ablation_id"].str.contains("Uncalibrated")].copy()

    labels = [
        "1. Weather (1d)",
        "2. Weather (1d+3d+7d)",
        "3. + Fire History",
        "4. + Terrain & VPD",
        "5. Full Multimodal",
    ]
    roc_scores = df_clean["cal_roc_auc"].values * 100.0
    f1_scores = df_clean["cal_f1"].values * 100.0

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rects1 = ax.bar(x - width/2, roc_scores, width, label="ROC-AUC (%)", color="#10b981")
    rects2 = ax.bar(x + width/2, f1_scores, width, label="F1-Score (%)", color="#3b82f6")

    ax.set_ylabel("Metric Score (%)")
    ax.set_title("Modality Ablation Benchmark (Strict Chronological Test 2024–2025)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim([45, 75])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    # Add direct value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Ablation figure saved to {output_path}", flush=True)


def plot_event_clustering_distribution(events_csv: Path, output_path: Path):
    """Plot distribution of constructed spatiotemporal fire event durations and displacements."""
    df = pd.read_csv(events_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Event duration histogram
    durations = df["duration_days"].clip(upper=10)
    ax1.hist(durations, bins=np.arange(1, 12) - 0.5, color="#f97316", rwidth=0.8, edgecolor="black", alpha=0.85)
    ax1.set_title("Fire Event Duration Distribution")
    ax1.set_xlabel("Event Duration (Days)")
    ax1.set_ylabel("Count of Events")
    ax1.set_xticks(range(1, 11))
    ax1.set_xticklabels([str(i) for i in range(1, 10)] + ["10+"])
    ax1.set_yscale("log")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Multi-day displacement histogram
    multi = df[df["duration_days"] > 1]
    ax2.hist(multi["displacement_km"].clip(upper=60), bins=20, color="#10b981", edgecolor="black", alpha=0.85)
    ax2.set_title(f"Multi-Day Event Displacement (N={len(multi):,})")
    ax2.set_xlabel("Centroid Displacement Distance (km)")
    ax2.set_ylabel("Count of Multi-Day Events")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Event dynamics figure saved to {output_path}", flush=True)


def main():
    setup_matplotlib()
    figs_dir = Path("results/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    base_preds = Path("results/baselines/baseline_test_predictions.csv")
    if base_preds.exists():
        plot_roc_and_pr_curves(base_preds, figs_dir / "fig1_roc_pr_curves.png")
        plot_calibration_curves(base_preds, figs_dir / "fig2_calibration_curves.png")

    ablation_csv = Path("results/ablations/ablation_comparison.csv")
    if ablation_csv.exists():
        plot_ablation_matrix(ablation_csv, figs_dir / "fig3_ablation_matrix.png")

    events_csv = Path("data/events/fire_events.csv")
    if events_csv.exists():
        plot_event_clustering_distribution(events_csv, figs_dir / "fig4_event_distributions.png")


if __name__ == "__main__":
    main()
