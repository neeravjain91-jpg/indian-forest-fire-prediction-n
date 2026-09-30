"""Probability calibration and out-of-distribution (OOD) uncertainty estimation."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ModelCalibrator:
    """Post-hoc probability calibrator supporting Raw (uncalibrated), Platt scaling, and Isotonic regression."""

    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        self.model = None

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> "ModelCalibrator":
        if self.method in ("raw", "none", "uncalibrated"):
            self.model = None
            return self

        probs = np.clip(np.asarray(val_probs, dtype=float), 1e-6, 1.0 - 1e-6)
        labels = np.asarray(val_labels, dtype=int)

        if self.method == "platt":
            # Logistic regression on log-odds
            log_odds = np.log(probs / (1.0 - probs)).reshape(-1, 1)
            self.model = LogisticRegression(solver="lbfgs", C=1.0)
            self.model.fit(log_odds, labels)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(probs, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.method in ("raw", "none", "uncalibrated") or self.model is None:
            return p

        if self.method == "platt":
            log_odds = np.log(p / (1.0 - p)).reshape(-1, 1)
            return self.model.predict_proba(log_odds)[:, 1]
        elif self.method == "isotonic":
            return self.model.predict(p)
        return p


def evaluate_calibration_methods(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compare Raw, Platt (Sigmoid), and Isotonic calibration protocols.
    
    Fits calibrators exclusively on validation data, and evaluates ECE, MCE,
    and Brier score loss on untouched test data.
    """
    from src.evaluation.metrics import compute_classification_metrics

    methods = ["raw", "platt", "isotonic"]
    results = {}

    for m in methods:
        calibrator = ModelCalibrator(method=m).fit(val_probs, val_labels)
        cal_test_p = calibrator.calibrate(test_probs)
        metrics = compute_classification_metrics(test_labels, cal_test_p)
        results[m] = metrics

    return results


class UncertaintyEstimator:
    """Epistemic uncertainty and distance-to-support Out-Of-Distribution (OOD) calculator."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.train_centroid = None
        self.train_std = None

    def fit(self, X_train: np.ndarray) -> "UncertaintyEstimator":
        X_scaled = self.scaler.fit_transform(X_train)
        self.train_centroid = np.mean(X_scaled, axis=0)
        self.train_std = np.std(X_scaled, axis=0) + 1e-6
        return self

    def compute_ood_distance(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized distance-to-support OOD score in feature space."""
        X_scaled = self.scaler.transform(X)
        diff = (X_scaled - self.train_centroid) / self.train_std
        distances = np.sqrt(np.sum(diff ** 2, axis=1)) / np.sqrt(X.shape[1])
        # Scale to [0, 1] range via sigmoid mapping
        ood_score = 1.0 / (1.0 + np.exp(-(distances - 2.5)))
        return np.round(ood_score, 4)

    def evaluate_uncertainty_error_correlation(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ood_scores: np.ndarray,
        n_deciles: int = 5,
    ) -> pd.DataFrame:
        """Verify whether prediction errors monotonically correlate with higher uncertainty."""
        y_t = np.asarray(y_true, dtype=int)
        y_pred = (np.asarray(y_prob, dtype=float) >= 0.5).astype(int)
        errors = (y_t != y_pred).astype(float)
        
        df = pd.DataFrame({"error": errors, "ood_score": ood_scores})
        df["uncertainty_bin"] = pd.qcut(df["ood_score"], q=n_deciles, duplicates="drop")
        
        summary = df.groupby("uncertainty_bin", observed=False).agg(
            sample_count=("error", "count"),
            mean_uncertainty=("ood_score", "mean"),
            error_rate=("error", "mean"),
        ).reset_index()
        summary["error_rate"] = summary["error_rate"].round(4)
        summary["mean_uncertainty"] = summary["mean_uncertainty"].round(4)
        return summary
