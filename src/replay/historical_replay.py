"""Historical Replay Engine for Wildfire Forecast Verification.

Given a historical origin time T:
1. Gathers strictly antecedent data available at T
2. Generates forward forecasts (T+24h)
3. Queries verified ground-truth active fires at T+24h
4. Computes spatial hit/miss/false-alarm metrics and event evolution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.events.event_clustering import haversine_km
from src.models.baselines import FEATURES_MULTIMODAL_39


class HistoricalReplayEngine:
    """Historical Replay Engine for prospective forecast vs actual verification."""

    def __init__(self, data_path: Path, model_path: Path, events_path: Path | None = None):
        print(f"Initializing Replay Engine from {data_path}...", flush=True)
        self.df = pd.read_csv(data_path)
        self.df["acq_date"] = pd.to_datetime(self.df["acq_date"])
        self.model = joblib.load(model_path)
        self.events_df = pd.read_csv(events_path) if events_path and events_path.exists() else None

    def execute_replay(
        self,
        date_str: str,
        probability_threshold: float = 0.5,
    ) -> dict:
        """Execute replay for a given origin date T."""
        t_date = pd.to_datetime(date_str)
        t_plus_24h = t_date + pd.Timedelta(days=1)

        # Observations available at origin T
        origin_obs = self.df[self.df["acq_date"] == t_date].copy()
        if len(origin_obs) == 0:
            # Fallback to nearest available date
            nearest_idx = (self.df["acq_date"] - t_date).abs().argmin()
            t_date = self.df.iloc[nearest_idx]["acq_date"]
            t_plus_24h = t_date + pd.Timedelta(days=1)
            origin_obs = self.df[self.df["acq_date"] == t_date].copy()

        # Generate forward forecast
        probs = self.model.predict_proba(origin_obs[FEATURES_MULTIMODAL_39])[:, 1]
        origin_obs["forecast_prob"] = np.round(probs, 4)
        origin_obs["forecast_risk"] = (probs >= probability_threshold).astype(int)

        # Verified actual observations at T+24h
        actual_obs = self.df[self.df["acq_date"] == t_plus_24h].copy()
        actual_fire_cells = set(
            zip(
                actual_obs[actual_obs["fire"] == 1]["grid_lat"].round(1),
                actual_obs[actual_obs["fire"] == 1]["grid_lon"].round(1),
            )
        )

        origin_obs["actual_fire_t24"] = [
            1 if (lat, lon) in actual_fire_cells else 0
            for lat, lon in zip(origin_obs["grid_lat"], origin_obs["grid_lon"])
        ]

        # Spatial verification stats
        y_pred = origin_obs["forecast_risk"].values
        y_true = origin_obs["actual_fire_t24"].values

        hits = int(np.sum((y_pred == 1) & (y_true == 1)))
        false_alarms = int(np.sum((y_pred == 1) & (y_true == 0)))
        misses = int(np.sum((y_pred == 0) & (y_true == 1)))
        correct_negatives = int(np.sum((y_pred == 0) & (y_true == 0)))

        prec = hits / max(1, hits + false_alarms)
        rec = hits / max(1, hits + misses)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec)

        # Active events at origin T
        active_events = []
        if self.events_df is not None:
            t_str = t_date.strftime("%Y-%m-%d")
            evts = self.events_df[
                (self.events_df["start_date"] <= t_str) & (self.events_df["end_date"] >= t_str)
            ].head(15)
            active_events = evts.to_dict(orient="records")

        # Top high-risk forecast cells
        high_risk_cells = (
            origin_obs[origin_obs["forecast_prob"] >= 0.40]
            .sort_values("forecast_prob", ascending=False)
            .head(50)[["grid_lat", "grid_lon", "forecast_prob", "actual_fire_t24", "temp_1d", "rh_1d", "vpd_1d", "elevation_m"]]
            .to_dict(orient="records")
        )

        return {
            "forecast_origin_date": t_date.strftime("%Y-%m-%d"),
            "verification_target_date": t_plus_24h.strftime("%Y-%m-%d"),
            "forecast_horizon": "T+24h (Next-Day)",
            "monitored_cells_count": len(origin_obs),
            "actual_fire_cells_count": len(actual_fire_cells),
            "forecast_hits": hits,
            "forecast_false_alarms": false_alarms,
            "forecast_misses": misses,
            "forecast_correct_negatives": correct_negatives,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "active_events_count": len(active_events),
            "active_events": active_events,
            "high_risk_forecast_sample": high_risk_cells,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/features/multimodal_features.csv")
    p.add_argument("--model", default="results/baselines/LightGBM.joblib")
    p.add_argument("--events", default="data/events/fire_events.csv")
    p.add_argument("--date", default="2024-03-25")
    p.add_argument("--output", default="results/replay_demo.json")
    args = p.parse_args()

    engine = HistoricalReplayEngine(Path(args.data), Path(args.model), Path(args.events))
    replay_result = engine.execute_replay(args.date)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(replay_result, f, indent=2)

    print(f"\n=== Historical Replay Execution: {replay_result['forecast_origin_date']} -> {replay_result['verification_target_date']} ===")
    print(f"Monitored Cells: {replay_result['monitored_cells_count']}")
    print(f"Hits: {replay_result['forecast_hits']} | False Alarms: {replay_result['forecast_false_alarms']} | Misses: {replay_result['forecast_misses']}")
    print(f"Precision: {replay_result['precision']:.2%} | Recall: {replay_result['recall']:.2%} | F1: {replay_result['f1_score']:.2%}")
    print(f"Active Events: {replay_result['active_events_count']}")


if __name__ == "__main__":
    main()
