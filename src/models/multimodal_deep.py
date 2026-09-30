"""Multimodal Spatiotemporal Deep Learning Network for Wildfire Forecasting.

Implements modular neural branches for:
1. Atmospheric & Multi-timescale Meteorology
2. Topography & Environmental Fuel Dryness
3. Fire History & Spatiotemporal Persistence
Fused via Gated Representation with Multi-Task Heads (Occurrence, Lead-24h, Persistence)
and Monte Carlo Dropout Epistemic Uncertainty Estimation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.calibration import ModelCalibrator, UncertaintyEstimator
from src.evaluation.metrics import compute_classification_metrics

# Modality Feature Definitions
WEATHER_COLS = [
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d",
    "temp_3d_mean", "temp_3d_max", "temp_3d_min", "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max", "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    "temp_7d_mean", "temp_7d_max", "temp_7d_min", "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max", "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]

ENV_TERRAIN_COLS = [
    "elevation_m", "slope_deg", "ruggedness_index",
    "vpd_1d", "vpd_3d_mean", "soil_drought_index",
]

HISTORY_COLS = [
    "grid_lat", "grid_lon", "hour",
    "fire_history_recurrence", "antecedent_fire_24h",
]


class ModalityBranch(nn.Module):
    """Feedforward encoder branch with LayerNorm and GELU activations."""

    def __init__(self, in_features: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFireNet(nn.Module):
    """Multimodal Spatiotemporal Deep Neural Network for Wildfire Forecasting."""

    def __init__(self, dropout: float = 0.15):
        super().__init__()
        # 1. Weather Branch (26 -> 32)
        self.weather_branch = ModalityBranch(len(WEATHER_COLS), 64, 32, dropout=dropout)
        
        # 2. Environment & Terrain Branch (6 -> 16)
        self.env_branch = ModalityBranch(len(ENV_TERRAIN_COLS), 32, 16, dropout=dropout)
        
        # 3. Fire History & Persistence Branch (5 -> 16)
        self.history_branch = ModalityBranch(len(HISTORY_COLS), 32, 16, dropout=dropout)

        fusion_dim = 32 + 16 + 16  # 64 dimensions

        # Gated Cross-Modality Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Multi-task Prediction Heads
        self.head_occurrence = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.head_lead24h = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.head_persistence = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_weather: torch.Tensor,
        x_env: torch.Tensor,
        x_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_w = self.weather_branch(x_weather)
        h_e = self.env_branch(x_env)
        h_h = self.history_branch(x_history)

        fused = torch.cat([h_w, h_e, h_h], dim=1)
        latent = self.fusion(fused) + fused  # Residual skip connection

        logits_occ = self.head_occurrence(latent).squeeze(-1)
        logits_lead24 = self.head_lead24h(latent).squeeze(-1)
        logits_pers = self.head_persistence(latent).squeeze(-1)

        return logits_occ, logits_lead24, logits_pers


def extract_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and convert dataframe subsets into torch float tensors."""
    w = torch.tensor(df[WEATHER_COLS].values, dtype=torch.float32)
    e = torch.tensor(df[ENV_TERRAIN_COLS].values, dtype=torch.float32)
    h = torch.tensor(df[HISTORY_COLS].values, dtype=torch.float32)
    return w, e, h


def train_multimodal_model(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    """Train the multimodal deep network, perform calibration and MC-dropout evaluation."""
    print("Loading datasets for multimodal training...", flush=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare inputs
    w_train, e_train, h_train = extract_tensors(train_df)
    y_train = torch.tensor(train_df["fire"].values, dtype=torch.float32)
    y_lead24_train = torch.tensor(train_df.get("target_fire_lead_24h", train_df["fire"]).values, dtype=torch.float32)
    y_pers_train = torch.tensor(train_df.get("target_event_persistence", train_df["fire"]).values, dtype=torch.float32)

    w_val, e_val, h_val = extract_tensors(val_df)
    y_val = torch.tensor(val_df["fire"].values, dtype=torch.float32)

    w_test, e_test, h_test = extract_tensors(test_df)
    y_test = test_df["fire"].values

    train_dataset = TensorDataset(w_train, e_train, h_train, y_train, y_lead24_train, y_pers_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MultimodalFireNet on device: {device}", flush=True)

    model = MultimodalFireNet(dropout=0.15).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_weights_path = output_dir / "multimodal_best_weights.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for bw, be, bh, by, by_lead, by_pers in train_loader:
            bw, be, bh = bw.to(device), be.to(device), bh.to(device)
            by, by_lead, by_pers = by.to(device), by_lead.to(device), by_pers.to(device)

            optimizer.zero_grad()
            l_occ, l_lead, l_pers = model(bw, be, bh)

            loss = bce(l_occ, by) + 0.5 * bce(l_lead, by_lead) + 0.5 * bce(l_pers, by_pers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)

        train_loss = total_loss / len(train_dataset)

        # Validation step
        model.eval()
        with torch.no_grad():
            vw, ve, vh = w_val.to(device), e_val.to(device), h_val.to(device)
            vl_occ, _, _ = model(vw, ve, vh)
            val_loss = bce(vl_occ, y_val.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

    # Load best model weights
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()

    # Fit calibration on validation set
    with torch.no_grad():
        val_probs = torch.sigmoid(model(w_val.to(device), e_val.to(device), h_val.to(device))[0]).cpu().numpy()
    calibrator = ModelCalibrator(method="isotonic")
    calibrator.fit(val_probs, val_df["fire"].values)

    # Test evaluation with Monte Carlo Dropout for Epistemic Uncertainty
    print("Evaluating test predictions with Monte Carlo Dropout...", flush=True)
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    model.apply(enable_dropout)
    mc_samples = 15
    mc_preds = []

    tw, te, th = w_test.to(device), e_test.to(device), h_test.to(device)
    with torch.no_grad():
        for _ in range(mc_samples):
            p = torch.sigmoid(model(tw, te, th)[0]).cpu().numpy()
            mc_preds.append(p)

    mc_preds = np.array(mc_preds)  # shape: (mc_samples, n_test)
    raw_mean_prob = np.mean(mc_preds, axis=0)
    epistemic_std = np.std(mc_preds, axis=0)  # Epistemic predictive variance
    calibrated_prob = calibrator.calibrate(raw_mean_prob)

    # Distance-to-support OOD evaluation
    all_features = WEATHER_COLS + ENV_TERRAIN_COLS + HISTORY_COLS
    ood_estimator = UncertaintyEstimator().fit(train_df[all_features].values)
    ood_scores = ood_estimator.compute_ood_distance(test_df[all_features].values)

    # Uncertainty error correlation
    unc_corr_df = ood_estimator.evaluate_uncertainty_error_correlation(
        y_test, calibrated_prob, ood_scores
    )
    unc_corr_df.to_csv(output_dir / "uncertainty_error_correlation.csv", index=False)

    uncal_metrics = compute_classification_metrics(y_test, raw_mean_prob)
    cal_metrics = compute_classification_metrics(y_test, calibrated_prob)

    results = {
        "model": "Multimodal_Spatiotemporal_Deep_Net",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "uncalibrated": uncal_metrics,
        "calibrated": cal_metrics,
        "mean_epistemic_std": float(round(np.mean(epistemic_std), 4)),
        "mean_ood_score": float(round(np.mean(ood_scores), 4)),
    }

    with open(output_dir / "multimodal_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save test predictions with uncertainty fields
    preds_df = pd.DataFrame({
        "grid_lat": test_df["grid_lat"],
        "grid_lon": test_df["grid_lon"],
        "acq_date": test_df["acq_date"],
        "y_true": y_test,
        "prob_raw": np.round(raw_mean_prob, 4),
        "prob_calibrated": np.round(calibrated_prob, 4),
        "epistemic_uncertainty": np.round(epistemic_std, 4),
        "ood_score": np.round(ood_scores, 4),
        "ecological_regime": test_df.get("ecological_regime", "UNKNOWN"),
    })
    preds_df.to_csv(output_dir / "multimodal_test_predictions.csv", index=False)

    print("\n=== Multimodal Deep Net (Calibrated Test Metrics on 2024-2025) ===")
    print(f"Accuracy:    {cal_metrics['accuracy']:.4f}")
    print(f"F1 Score:    {cal_metrics['f1']:.4f}")
    print(f"ROC-AUC:     {cal_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:      {cal_metrics['pr_auc']:.4f}")
    print(f"Brier Score: {cal_metrics['brier_score']:.4f}")
    print(f"ECE:         {cal_metrics['ece']:.4f}")
    print(f"Mean Epistemic Uncertainty: {results['mean_epistemic_std']:.4f}")

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train_chronological.csv")
    p.add_argument("--val", default="data/splits/val_chronological.csv")
    p.add_argument("--test", default="data/splits/test_chronological.csv")
    p.add_argument("--output-dir", default="results/multimodal")
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()

    train_multimodal_model(
        Path(args.train),
        Path(args.val),
        Path(args.test),
        Path(args.output_dir),
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
