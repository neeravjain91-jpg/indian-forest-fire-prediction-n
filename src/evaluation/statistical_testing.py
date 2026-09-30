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


def compute_factorial_interaction_bootstrap(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    prob_c: np.ndarray,
    prob_d: np.ndarray,
    metric_name: str = "roc_auc",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float | bool]]:
    """Calculate 2x2 factorial effects and interaction via paired bootstrap resampling.
    
    Factorial Structure:
      A = HGB + 31 Baseline
      B = HGB + 39 Multimodal
      C = LightGBM + 31 Baseline
      D = LightGBM + 39 Multimodal

    Quantified Effects:
      1. feature_effect_hgb: B - A
      2. feature_effect_lgbm: D - C
      3. feature_main_effect: ((B - A) + (D - C)) / 2
      4. model_effect_31: C - A
      5. model_effect_39: D - B
      6. model_main_effect: ((C - A) + (D - B)) / 2
      7. interaction: (D - C) - (B - A)
    """
    y_t = np.asarray(y_true, dtype=int)
    p_a = np.asarray(prob_a, dtype=float)
    p_b = np.asarray(prob_b, dtype=float)
    p_c = np.asarray(prob_c, dtype=float)
    p_d = np.asarray(prob_d, dtype=float)
    n = len(y_t)

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

    # Point estimates on observed test set
    mA = calc_metric(y_t, p_a)
    mB = calc_metric(y_t, p_b)
    mC = calc_metric(y_t, p_c)
    mD = calc_metric(y_t, p_d)

    obs_effects = {
        "feature_effect_hgb": mB - mA,
        "feature_effect_lgbm": mD - mC,
        "feature_main_effect": ((mB - mA) + (mD - mC)) / 2.0,
        "model_effect_31": mC - mA,
        "model_effect_39": mD - mB,
        "model_main_effect": ((mC - mA) + (mD - mB)) / 2.0,
        "interaction": (mD - mC) - (mB - mA),
    }

    boot_effects = {k: [] for k in obs_effects}
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        by = y_t[boot_idx]
        if len(np.unique(by)) < 2:
            continue
        bA = calc_metric(by, p_a[boot_idx])
        bB = calc_metric(by, p_b[boot_idx])
        bC = calc_metric(by, p_c[boot_idx])
        bD = calc_metric(by, p_d[boot_idx])

        fe_hgb = bB - bA
        fe_lgb = bD - bC
        boot_effects["feature_effect_hgb"].append(fe_hgb)
        boot_effects["feature_effect_lgbm"].append(fe_lgb)
        boot_effects["feature_main_effect"].append((fe_hgb + fe_lgb) / 2.0)

        me_31 = bC - bA
        me_39 = bD - bB
        boot_effects["model_effect_31"].append(me_31)
        boot_effects["model_effect_39"].append(me_39)
        boot_effects["model_main_effect"].append((me_31 + me_39) / 2.0)

        boot_effects["interaction"].append(fe_lgb - fe_hgb)

    summary = {}
    for eff_name, vals in boot_effects.items():
        arr = np.array(vals)
        ci_low = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        ex_zero = bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
        summary[eff_name] = {
            "metric": metric_name,
            "effect_name": eff_name,
            "observed": round(float(obs_effects[eff_name]), 4),
            "ci_95_lower": round(ci_low, 4),
            "ci_95_upper": round(ci_high, 4),
            "ci_excludes_zero": ex_zero,
        }

    return summary
