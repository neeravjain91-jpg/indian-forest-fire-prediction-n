"""Statistical hypothesis testing, bootstrap confidence intervals, and rare-event metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_precision_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_list: list[int] = [100, 250, 500, 1000],
) -> pd.DataFrame:
    """Calculate Precision@k and Recall@k for imbalanced rare-event forecasting targets.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Model predicted probabilities.
    k_list : list[int]
        Cutoff thresholds for top-k predicted high-risk cells.
        
    Returns
    -------
    pd.DataFrame
        Table with k, hits, precision_at_k, recall_at_k.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    total_positives = int(np.sum(y_t))
    n = len(y_t)

    sorted_indices = np.argsort(-y_p)
    y_t_sorted = y_t[sorted_indices]

    results = []
    for k in k_list:
        k_eff = min(k, n)
        top_k_labels = y_t_sorted[:k_eff]
        hits = int(np.sum(top_k_labels))
        prec_k = hits / float(k_eff) if k_eff > 0 else 0.0
        rec_k = hits / float(total_positives) if total_positives > 0 else 0.0

        results.append({
            "k": k_eff,
            "hits": hits,
            "total_positives": total_positives,
            "precision_at_k": round(prec_k, 4),
            "recall_at_k": round(rec_k, 4),
        })

    return pd.DataFrame(results)


def compute_bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_name: str = "roc_auc",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Compute non-parametric bootstrap 95% Confidence Interval for metric difference.
    
    Delta = Metric(Model A) - Metric(Model B)
    """
    y_t = np.asarray(y_true, dtype=int)
    p_a = np.asarray(y_prob_a, dtype=float)
    p_b = np.asarray(y_prob_b, dtype=float)
    n = len(y_t)

    rng = np.random.default_rng(seed)
    deltas = []

    def calc_metric(y, p):
        if metric_name == "roc_auc":
            return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        elif metric_name == "pr_auc":
            return average_precision_score(y, p) if len(np.unique(y)) > 1 else 0.0
        elif metric_name == "brier":
            return brier_score_loss(y, p)
        elif metric_name == "f1":
            pred = (p >= 0.5).astype(int)
            return f1_score(y, pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    base_a = calc_metric(y_t, p_a)
    base_b = calc_metric(y_t, p_b)
    obs_delta = base_a - base_b

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        by = y_t[boot_idx]
        if len(np.unique(by)) < 2:
            continue
        m_a = calc_metric(by, p_a[boot_idx])
        m_b = calc_metric(by, p_b[boot_idx])
        deltas.append(m_a - m_b)

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    excludes_zero = bool((ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0))

    return {
        "metric": metric_name,
        "observed_delta": round(float(obs_delta), 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "ci_excludes_zero": excludes_zero,
    }
