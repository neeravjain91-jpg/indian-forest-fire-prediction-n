"""Historical Replay Engine & Multi-Date Retrospective Verification Benchmark.

Evaluates forward forecasting (T -> T+24h) across multiple valid historical origin
dates sampled systematically across the held-out test period (2024-2025).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

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
        """Execute replay for a single forecast origin date T."""
        t_date = pd.to_datetime(date_str)
        t_plus_24h = t_date + pd.Timedelta(days=1)

        origin_obs = self.df[self.df["acq_date"] == t_date].copy()
        if len(origin_obs) == 0:
            nearest_idx = (self.df["acq_date"] - t_date).abs().argmin()
            t_date = self.df.iloc[nearest_idx]["acq_date"]
            t_plus_24h = t_date + pd.Timedelta(days=1)
            origin_obs = self.df[self.df["acq_date"] == t_date].copy()

        probs = self.model.predict_proba(origin_obs[FEATURES_MULTIMODAL_39])[:, 1]
        origin_obs["forecast_prob"] = np.round(probs, 4)
        origin_obs["forecast_risk"] = (probs >= probability_threshold).astype(int)

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

        y_pred = origin_obs["forecast_risk"].values
        y_true = origin_obs["actual_fire_t24"].values

        hits = int(np.sum((y_pred == 1) & (y_true == 1)))
        false_alarms = int(np.sum((y_pred == 1) & (y_true == 0)))
        misses = int(np.sum((y_pred == 0) & (y_true == 1)))
        correct_negatives = int(np.sum((y_pred == 0) & (y_true == 0)))

        prec = hits / max(1, hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
        rec = hits / max(1, hits + misses) if (hits + misses) > 0 else 0.0
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0

        active_events = []
        if self.events_df is not None:
            t_str = t_date.strftime("%Y-%m-%d")
            evts = self.events_df[
                (self.events_df["start_date"] <= t_str) & (self.events_df["end_date"] >= t_str)
            ].head(15)
            active_events = evts.to_dict(orient="records")

        return {
            "forecast_origin_date": t_date.strftime("%Y-%m-%d"),
            "verification_target_date": t_plus_24h.strftime("%Y-%m-%d"),
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
        }

    def run_multi_date_benchmark(
        self,
        output_dir: Path,
        sample_dates: list[str] | None = None,
        probability_threshold: float = 0.40,
    ) -> pd.DataFrame:
        """Run systematic historical replay benchmark across multiple dates."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if sample_dates is None:
            # 20 representative dates across 2024-2025 test years, capturing active fire season (Feb-May)
            sample_dates = [
                "2024-02-15", "2024-02-28", "2024-03-10", "2024-03-20", "2024-03-30",
                "2024-04-10", "2024-04-20", "2024-04-30", "2024-05-10", "2024-05-25",
                "2025-02-15", "2025-02-28", "2025-03-10", "2025-03-20", "2025-03-30",
                "2025-04-10", "2025-04-20", "2025-04-30", "2025-05-10", "2025-05-25",
            ]

        results = []
        for d in sample_dates:
            res = self.execute_replay(d, probability_threshold=probability_threshold)
            results.append({
                "origin_date": res["forecast_origin_date"],
                "target_date": res["verification_target_date"],
                "monitored_cells": res["monitored_cells_count"],
                "actual_fires": res["actual_fire_cells_count"],
                "hits": res["forecast_hits"],
                "false_alarms": res["forecast_false_alarms"],
                "misses": res["forecast_misses"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1_score": res["f1_score"],
                "active_events": res["active_events_count"],
            })

        bench_df = pd.DataFrame(results)
        bench_df.to_csv(output_dir / "multi_date_historical_benchmark.csv", index=False)

        macro_summary = {
            "n_dates_evaluated": len(bench_df),
            "macro_precision": round(float(bench_df["precision"].mean()), 4),
            "macro_recall": round(float(bench_df["recall"].mean()), 4),
            "macro_f1": round(float(bench_df["f1_score"].mean()), 4),
            "total_monitored_cells": int(bench_df["monitored_cells"].sum()),
            "total_hits": int(bench_df["hits"].sum()),
            "total_false_alarms": int(bench_df["false_alarms"].sum()),
            "total_misses": int(bench_df["misses"].sum()),
        }

        with open(output_dir / "multi_date_benchmark_summary.json", "w", encoding="utf-8") as f:
            json.dump(macro_summary, f, indent=2)

        print("\n=== Multi-Date Historical Replay Benchmark (N=20 Dates in 2024-2025) ===")
        print(bench_df[["origin_date", "monitored_cells", "hits", "false_alarms", "misses", "precision", "recall", "f1_score"]].to_string(index=False))
        print(f"\nMacro Averages: Precision={macro_summary['macro_precision']:.2%}, Recall={macro_summary['macro_recall']:.2%}, F1={macro_summary['macro_f1']:.2%}")
        return bench_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/features/multimodal_features.csv")
    p.add_argument("--model", default="results/baselines/ExpD_LGBM_39_Multimodal.joblib")
    p.add_argument("--events", default="data/events/fire_events.csv")
    p.add_argument("--output-dir", default="results/replay")
    args = p.parse_args()

    engine = HistoricalReplayEngine(Path(args.data), Path(args.model), Path(args.events))
    engine.run_multi_date_benchmark(Path(args.output_dir))


if __name__ == "__main__":
    main()
