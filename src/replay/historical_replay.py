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

        req_features = getattr(self.model, "feature_names_in_", None)
        if req_features is not None:
            features_to_use = list(req_features)
        elif hasattr(self.model, "n_features_in_") and self.model.n_features_in_ == 31:
            from src.models.baselines import FEATURES_BASELINE_31
            features_to_use = FEATURES_BASELINE_31
        else:
            features_to_use = FEATURES_MULTIMODAL_39

        for feat in features_to_use:
            if feat not in origin_obs.columns:
                if feat == "elevation_m":
                    origin_obs[feat] = 350.0
                elif feat == "slope_deg":
                    origin_obs[feat] = 0.5
                elif feat == "ruggedness_index":
                    origin_obs[feat] = 50.0
                elif feat in ("vpd_1d", "vpd_3d_mean"):
                    origin_obs[feat] = 2.0
                elif feat == "soil_drought_index":
                    origin_obs[feat] = 0.4
                elif feat == "fire_history_recurrence":
                    origin_obs[feat] = 0.1
                elif feat == "antecedent_fire_24h":
                    origin_obs[feat] = 0.0
                else:
                    origin_obs[feat] = 0.0

        probs = self.model.predict_proba(origin_obs[features_to_use])[:, 1]
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
        misses_in_candidate = int(np.sum((y_pred == 0) & (y_true == 1)))
        correct_negatives = int(np.sum((y_pred == 0) & (y_true == 0)))

        target_fires_total = len(actual_fire_cells)
        target_fires_in_candidate = int(np.sum(y_true == 1))
        misses_outside_candidate = max(0, target_fires_total - target_fires_in_candidate)
        total_domain_misses = misses_in_candidate + misses_outside_candidate

        # 1. Candidate-Domain Metrics (conditioned on evaluated candidate cells)
        candidate_prec = hits / max(1, hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
        candidate_rec = hits / max(1, target_fires_in_candidate) if target_fires_in_candidate > 0 else 0.0
        candidate_f1 = (2 * candidate_prec * candidate_rec) / max(1e-6, candidate_prec + candidate_rec) if (candidate_prec + candidate_rec) > 0 else 0.0

        # 2. Full Spatial Domain Recall (treating unmonitored target fire cells as misses)
        full_spatial_rec = hits / max(1, target_fires_total) if target_fires_total > 0 else 0.0
        full_spatial_f1 = (2 * candidate_prec * full_spatial_rec) / max(1e-6, candidate_prec + full_spatial_rec) if (candidate_prec + full_spatial_rec) > 0 else 0.0

        fpr = false_alarms / max(1, correct_negatives + false_alarms) if (correct_negatives + false_alarms) > 0 else 0.0

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
            "candidate_cells_count": len(origin_obs),
            "actual_fire_cells_total": target_fires_total,
            "target_fires_in_candidate": target_fires_in_candidate,
            "forecast_hits": hits,
            "forecast_false_alarms": false_alarms,
            "misses_in_candidate": misses_in_candidate,
            "misses_outside_candidate": misses_outside_candidate,
            "total_domain_misses": total_domain_misses,
            "forecast_correct_negatives": correct_negatives,
            "precision": round(candidate_prec, 4),
            "candidate_domain_recall": round(candidate_rec, 4),
            "candidate_domain_f1": round(candidate_f1, 4),
            "full_spatial_recall": round(full_spatial_rec, 4),
            "full_spatial_f1": round(full_spatial_f1, 4),
            "false_positive_rate": round(fpr, 4),
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
                "candidate_cells": res["candidate_cells_count"],
                "actual_fires_total": res["actual_fire_cells_total"],
                "actual_fires_in_candidate": res["target_fires_in_candidate"],
                "hits": res["forecast_hits"],
                "false_alarms": res["forecast_false_alarms"],
                "misses_in_candidate": res["misses_in_candidate"],
                "misses_outside_candidate": res["misses_outside_candidate"],
                "total_misses": res["total_domain_misses"],
                "precision": res["precision"],
                "candidate_recall": res["candidate_domain_recall"],
                "candidate_f1": res["candidate_domain_f1"],
                "full_spatial_recall": res["full_spatial_recall"],
                "full_spatial_f1": res["full_spatial_f1"],
                "false_positive_rate": res["false_positive_rate"],
            })

        bench_df = pd.DataFrame(results)
        bench_df.to_csv(output_dir / "multi_date_historical_benchmark.csv", index=False)

        macro_summary = {
            "n_dates_evaluated": len(bench_df),
            "macro_precision": round(float(bench_df["precision"].mean()), 4),
            "macro_candidate_recall": round(float(bench_df["candidate_recall"].mean()), 4),
            "macro_candidate_f1": round(float(bench_df["candidate_f1"].mean()), 4),
            "macro_full_spatial_recall": round(float(bench_df["full_spatial_recall"].mean()), 4),
            "macro_full_spatial_f1": round(float(bench_df["full_spatial_f1"].mean()), 4),
            "macro_false_positive_rate": round(float(bench_df["false_positive_rate"].mean()), 4),
            "total_candidate_cells": int(bench_df["candidate_cells"].sum()),
            "total_actual_fires": int(bench_df["actual_fires_total"].sum()),
            "total_hits": int(bench_df["hits"].sum()),
            "total_false_alarms": int(bench_df["false_alarms"].sum()),
        }

        with open(output_dir / "multi_date_benchmark_summary.json", "w", encoding="utf-8") as f:
            json.dump(macro_summary, f, indent=2)

        print("\n=== Multi-Date Historical Replay Benchmark (N=20 Dates in 2024-2025) ===")
        print(bench_df[["origin_date", "candidate_cells", "hits", "false_alarms", "precision", "candidate_recall", "full_spatial_recall"]].to_string(index=False))
        print(f"\nMacro Averages: Precision={macro_summary['macro_precision']:.2%}, Candidate-Recall={macro_summary['macro_candidate_recall']:.2%}, Full-Spatial-Recall={macro_summary['macro_full_spatial_recall']:.2%}")
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
