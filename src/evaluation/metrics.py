"""Evaluation metrics for wildfire classification, probabilistic calibration, and regional assessment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width probability bins.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_t)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (y_p >= bin_lower) & (y_p <= bin_upper if i == n_bins - 1 else y_p < bin_upper)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate comprehensive classification and probabilistic calibration metrics."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_p >= threshold).astype(int)

    has_both_classes = len(np.unique(y_t)) > 1

    roc_auc = float(roc_auc_score(y_t, y_p)) if has_both_classes else float("nan")
    pr_auc = float(average_precision_score(y_t, y_p)) if has_both_classes else float("nan")
    brier = float(brier_score_loss(y_t, y_p))
    ece = compute_expected_calibration_error(y_t, y_p, n_bins=10)

    return {
        "accuracy": round(float(accuracy_score(y_t, y_pred)), 4),
        "precision": round(float(precision_score(y_t, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_t, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_t, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
    }


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Generate reliability diagram data points."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    rows = []
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (y_p >= bin_lower) & (y_p <= bin_upper if i == n_bins - 1 else y_p < bin_upper)
        count = int(np.sum(mask))
        if count > 0:
            mean_conf = float(np.mean(y_p[mask]))
            empirical_freq = float(np.mean(y_t[mask]))
        else:
            mean_conf = float((bin_lower + bin_upper) / 2.0)
            empirical_freq = float("nan")
        rows.append({
            "bin_lower": round(bin_lower, 2),
            "bin_upper": round(bin_upper, 2),
            "bin_center": round((bin_lower + bin_upper) / 2.0, 2),
            "mean_confidence": round(mean_conf, 4),
            "empirical_frequency": round(empirical_freq, 4) if not np.isnan(empirical_freq) else 0.0,
            "sample_count": count,
        })
    return pd.DataFrame(rows)


def evaluate_regional_breakdown(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    region_col: str = "ecological_regime",
) -> pd.DataFrame:
    """Evaluate performance metrics across distinct ecological and geographic regimes."""
    records = []
    for reg, grp in df.groupby(region_col):
        if len(grp) < 10:
            continue
        y_t = grp[y_true_col].values
        y_p = grp[y_prob_col].values
        m = compute_classification_metrics(y_t, y_p)
        records.append({
            "regime": reg,
            "sample_count": len(grp),
            "fire_count": int(np.sum(y_t)),
            "fire_rate": round(float(np.mean(y_t)), 4),
            **m,
        })
    return pd.DataFrame(records).sort_values("sample_count", ascending=False).reset_index(drop=True)
