"""Multimodal Spatiotemporal Deep Learning Network with Multi-Scale Temporal Feature Encoders.

Architecture:
1. Multi-Scale Temporal Weather Encoder: 2-layer Bidirectional GRU over ordered temporal multi-scale
   representation [t_{-7d}, t_{-3d}, t_{-1d}] capturing antecedent atmospheric drying trends.
2. Topography & Environment Encoder: MLP over authoritative DEM elevation, slope, TRI, and VPD.
3. Fire History & Spatial Persistence Encoder: MLP over causal recurrence, antecedent fire, coordinates.
4. Gated Cross-Modality Fusion Layer with Residual Skip Connection.
5. Multi-Task Heads: Diagnostic Occurrence (T), Forward Lead (T+24h), Connected Event Persistence.
6. Epistemic Uncertainty via Monte Carlo Dropout with empirical error correlation analysis.

Scientific Note:
The input sequence consists of ordered multi-scale temporal summaries [7d mean, 3d mean, 1d observation]
rather than continuous hourly weather trajectories. It functions as a multi-timescale sequential representation
encoder across synoptic and immediate timescales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.calibration import ModelCalibrator, UncertaintyEstimator
from src.evaluation.metrics import compute_classification_metrics

# Modality Column Definitions
ENV_TERRAIN_COLS = [
    "elevation_m", "slope_deg", "ruggedness_index",
    "vpd_1d", "vpd_3d_mean", "soil_drought_index",
]

HISTORY_COLS = [
    "grid_lat", "grid_lon", "hour",
    "fire_history_recurrence", "antecedent_fire_24h",
]


class TemporalWeatherEncoder(nn.Module):
    """Bidirectional GRU encoding the ordered multi-scale sequence [7d, 3d, 1d]."""

    def __init__(self, in_features: int = 6, hidden_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq shape: (batch_size, 3, 6)
        out, _ = self.gru(x_seq)
        # Take the final temporal representation
        last_step = out[:, -1, :]
        return self.layer_norm(last_step)


class StaticBranch(nn.Module):
    """Feedforward encoder for static terrain and fire history features."""

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


class SpatiotemporalMultimodalFireNet(nn.Module):
    """Deep Spatiotemporal Multimodal Network for Wildfire Forecasting."""

    def __init__(self, dropout: float = 0.15):
        super().__init__()
        # 1. Temporal Weather Encoder (BiGRU: 3x6 -> 48)
        self.weather_encoder = TemporalWeatherEncoder(in_features=6, hidden_dim=24, dropout=dropout)
        
        # 2. Real DEM Terrain & Environment Encoder (6 -> 16)
        self.env_encoder = StaticBranch(len(ENV_TERRAIN_COLS), 32, 16, dropout=dropout)
        
        # 3. Fire History Encoder (5 -> 16)
        self.history_encoder = StaticBranch(len(HISTORY_COLS), 32, 16, dropout=dropout)

        fusion_dim = 48 + 16 + 16  # 80 dimensions

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
        x_weather_seq: torch.Tensor,
        x_env: torch.Tensor,
        x_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_w = self.weather_encoder(x_weather_seq)
        h_e = self.env_encoder(x_env)
        h_h = self.history_encoder(x_history)

        fused = torch.cat([h_w, h_e, h_h], dim=1)
        latent = self.fusion(fused) + fused

        logits_occ = self.head_occurrence(latent).squeeze(-1)
        logits_lead24 = self.head_lead24h(latent).squeeze(-1)
        logits_pers = self.head_persistence(latent).squeeze(-1)

        return logits_occ, logits_lead24, logits_pers


def build_weather_sequences(df: pd.DataFrame) -> torch.Tensor:
    """Build ordered 3-step temporal sequence tensor: [7d, 3d, 1d]."""
    # Step 1: 7-day antecedent
    s7 = df[["temp_7d_mean", "rh_7d_mean", "wind_7d_mean", "pressure_7d_mean", "soil_7d_mean", "rain_7d_total"]].copy()
    s7["rain_7d_total"] /= 7.0
    
    # Step 2: 3-day intermediate
    s3 = df[["temp_3d_mean", "rh_3d_mean", "wind_3d_mean", "pressure_3d_mean", "soil_3d_mean", "rain_3d_total"]].copy()
    s3["rain_3d_total"] /= 3.0
    
    # Step 3: 1-day instantaneous
    s1 = df[["temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d"]].copy()

    # Stack along time dimension: shape (N, 3, 6)
    arr7 = s7.values[:, np.newaxis, :]
    arr3 = s3.values[:, np.newaxis, :]
    arr1 = s1.values[:, np.newaxis, :]
    seq = np.concatenate([arr7, arr3, arr1], axis=1)
    return torch.tensor(seq, dtype=torch.float32)


def train_temporal_multimodal_model(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    """Train temporal BiGRU multimodal network with MC Dropout and uncertainty correlation."""
    print("Loading datasets for temporal multimodal deep model...", flush=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build sequence and static inputs
    w_train_seq = build_weather_sequences(train_df)
    e_train = torch.tensor(train_df[ENV_TERRAIN_COLS].values, dtype=torch.float32)
    h_train = torch.tensor(train_df[HISTORY_COLS].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["fire"].values, dtype=torch.float32)
    y_lead_train = torch.tensor(train_df["target_fire_lead_24h"].values, dtype=torch.float32)
    y_pers_train = torch.tensor(train_df["target_event_persistence"].values, dtype=torch.float32)

    w_val_seq = build_weather_sequences(val_df)
    e_val = torch.tensor(val_df[ENV_TERRAIN_COLS].values, dtype=torch.float32)
    h_val = torch.tensor(val_df[HISTORY_COLS].values, dtype=torch.float32)
    y_val = torch.tensor(val_df["fire"].values, dtype=torch.float32)

    w_test_seq = build_weather_sequences(test_df)
    e_test = torch.tensor(test_df[ENV_TERRAIN_COLS].values, dtype=torch.float32)
    h_test = torch.tensor(test_df[HISTORY_COLS].values, dtype=torch.float32)
    y_test = test_df["fire"].values

    train_dataset = TensorDataset(w_train_seq, e_train, h_train, y_train, y_lead_train, y_pers_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training SpatiotemporalMultimodalFireNet on device: {device}", flush=True)

    model = SpatiotemporalMultimodalFireNet(dropout=0.15).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_weights_path = output_dir / "temporal_multimodal_best_weights.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for bw_seq, be, bh, by, by_lead, by_pers in train_loader:
            bw_seq, be, bh = bw_seq.to(device), be.to(device), bh.to(device)
            by, by_lead, by_pers = by.to(device), by_lead.to(device), by_pers.to(device)

            optimizer.zero_grad()
            l_occ, l_lead, l_pers = model(bw_seq, be, bh)

            loss = bce(l_occ, by) + 0.5 * bce(l_lead, by_lead) + 0.5 * bce(l_pers, by_pers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)

        train_loss = total_loss / len(train_dataset)

        # Validation step
        model.eval()
        with torch.no_grad():
            vw, ve, vh = w_val_seq.to(device), e_val.to(device), h_val.to(device)
            vl_occ, _, _ = model(vw, ve, vh)
            val_loss = bce(vl_occ, y_val.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

    # Load best weights
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()

    # Isotonic calibration on validation predictions
    with torch.no_grad():
        val_probs = torch.sigmoid(model(w_val_seq.to(device), e_val.to(device), h_val.to(device))[0]).cpu().numpy()
    calibrator = ModelCalibrator(method="isotonic").fit(val_probs, val_df["fire"].values)

    # Monte Carlo Dropout for Epistemic Uncertainty
    print("Evaluating test predictions with Monte Carlo Dropout...", flush=True)
    def enable_dropout(m):
        if type(m) in (nn.Dropout, nn.GRU):
            m.train()

    model.apply(enable_dropout)
    mc_samples = 15
    mc_preds = []

    tw, te, th = w_test_seq.to(device), e_test.to(device), h_test.to(device)
    with torch.no_grad():
        for _ in range(mc_samples):
            p = torch.sigmoid(model(tw, te, th)[0]).cpu().numpy()
            mc_preds.append(p)

    mc_preds = np.array(mc_preds)
    raw_mean_prob = np.mean(mc_preds, axis=0)
    epistemic_std = np.std(mc_preds, axis=0)
    calibrated_prob = calibrator.calibrate(raw_mean_prob)

    # Evaluate correlation between epistemic uncertainty and empirical error
    empirical_error = np.abs(y_test - calibrated_prob)
    spearman_corr, spearman_pval = spearmanr(epistemic_std, empirical_error)

    # Quantile analysis
    unc_df = pd.DataFrame({
        "y_true": y_test,
        "prob": calibrated_prob,
        "epistemic_std": epistemic_std,
        "error": empirical_error,
    })
    unc_df["unc_quantile"] = pd.qcut(unc_df["epistemic_std"], q=5, labels=["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"])
    quantile_summary = unc_df.groupby("unc_quantile", observed=False).agg(
        sample_count=("error", "count"),
        mean_uncertainty=("epistemic_std", "mean"),
        mean_error=("error", "mean"),
    ).reset_index()

    quantile_summary.to_csv(output_dir / "uncertainty_quantile_validation.csv", index=False)

    uncal_m = compute_classification_metrics(y_test, raw_mean_prob)
    cal_m = compute_classification_metrics(y_test, calibrated_prob)

    results = {
        "model": "SpatiotemporalMultimodalFireNet_BiGRU",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "uncalibrated": uncal_m,
        "calibrated": cal_m,
        "mean_epistemic_std": float(round(np.mean(epistemic_std), 4)),
        "spearman_corr_uncertainty_error": float(round(spearman_corr, 4)),
        "spearman_pval": float(spearman_pval),
    }

    with open(output_dir / "temporal_multimodal_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    preds_df = pd.DataFrame({
        "grid_lat": test_df["grid_lat"],
        "grid_lon": test_df["grid_lon"],
        "acq_date": test_df["acq_date"],
        "y_true": y_test,
        "prob_raw": np.round(raw_mean_prob, 4),
        "prob_calibrated": np.round(calibrated_prob, 4),
        "epistemic_uncertainty": np.round(epistemic_std, 4),
        "ecological_regime": test_df.get("ecological_regime", "UNKNOWN"),
    })
    preds_df.to_csv(output_dir / "temporal_multimodal_test_predictions.csv", index=False)

    print("\n=== Temporal BiGRU Multimodal Model (Calibrated Test Metrics 2024-2025) ===")
    print(f"Accuracy:    {cal_m['accuracy']:.4f}")
    print(f"F1 Score:    {cal_m['f1']:.4f}")
    print(f"ROC-AUC:     {cal_m['roc_auc']:.4f}")
    print(f"PR-AUC:      {cal_m['pr_auc']:.4f}")
    print(f"Brier Score: {cal_m['brier_score']:.4f}")
    print(f"ECE:         {cal_m['ece']:.4f}")
    print(f"Uncertainty-Error Spearman r: {spearman_corr:.4f} (p = {spearman_pval:.2e})")

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train_chronological.csv")
    p.add_argument("--val", default="data/splits/val_chronological.csv")
    p.add_argument("--test", default="data/splits/test_chronological.csv")
    p.add_argument("--output-dir", default="results/multimodal")
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()

    train_temporal_multimodal_model(
        Path(args.train),
        Path(args.val),
        Path(args.test),
        Path(args.output_dir),
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
