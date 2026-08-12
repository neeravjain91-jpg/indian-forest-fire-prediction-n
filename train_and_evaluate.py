from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover",
    "soil_moisture_0_to_7cm",
]
HISTORY = ["rain_24h", "rain_72h", "rain_168h", "avg_temp_24h", "avg_humidity_24h", "max_wind_24h"]


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 5),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "positive_rate": round(float(np.mean(y)), 5),
    }


def models():
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=12, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]),
    }


def fit_score(train, test, features):
    Xtr, ytr = train[features], train.fire.astype(int)
    Xte, yte = test[features], test.fire.astype(int)
    result = {}
    for name, model in models().items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        result[name] = metrics(yte, p)
    return result


def temporal_experiment(df, features):
    years = sorted(df.year.unique())
    if not {2018, 2023, 2024, 2025}.issubset(set(years)):
        raise ValueError(f"Expected 2018–2025 data; found years {years}")
    train = df[df.year <= 2023]
    validation = df[df.year == 2024]
    test = df[df.year == 2025]
    validation_scores = fit_score(train, validation, features)
    # Select the model by validation PR-AUC, then refit on all pre-test years.
    selected = max(validation_scores, key=lambda k: validation_scores[k]["pr_auc"])
    final_model = models()[selected]
    final_model.fit(df[df.year <= 2024][features], df[df.year <= 2024].fire.astype(int))
    p = final_model.predict_proba(test[features])[:, 1]
    return {
        "train_years": [2018, 2019, 2020, 2021, 2022, 2023],
        "validation_year": 2024,
        "test_year": 2025,
        "validation": validation_scores,
        "selected_model": selected,
        "future_test": metrics(test.fire.astype(int), p),
    }


def spatial_experiment(df, features, seed=42):
    # 2-degree geographic blocks create a group split that keeps nearby cells together.
    work = df.copy()
    work["spatial_group"] = (work.grid_lat // 2).astype(int).astype(str) + "_" + (work.grid_lon // 2).astype(int).astype(str)
    groups = np.array(sorted(work.spatial_group.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    cut = max(1, int(round(len(groups) * 0.20)))
    test_groups = set(groups[:cut])
    train = work[~work.spatial_group.isin(test_groups)]
    test = work[work.spatial_group.isin(test_groups)]
    return {
        "group_definition": "2-degree latitude/longitude blocks",
        "train_groups": int(len(set(train.spatial_group))),
        "test_groups": int(len(test_groups)),
        "test_fraction": round(len(test) / len(work), 5),
        "scores": fit_score(train, test, features),
    }


def ablation(df, seed=42):
    train = df[df.year <= 2023]
    test = df[df.year == 2025]
    feature_sets = {
        "instantaneous_only": BASE,
        "plus_24h": BASE + ["rain_24h", "avg_temp_24h", "avg_humidity_24h", "max_wind_24h"],
        "plus_24_72_168h_rain": BASE + ["rain_24h", "rain_72h", "rain_168h"],
        "full_multiscale": BASE + HISTORY,
    }
    rows = []
    for name, features in feature_sets.items():
        scores = fit_score(train, test, features)
        for model, values in scores.items():
            rows.append({"feature_set": name, "model": model, **values})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results")
    args = p.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["acq_date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    features = BASE + HISTORY
    missing = [c for c in features + ["fire", "grid_lat", "grid_lon", "year"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    report = {
        "dataset": {
            "rows": int(len(df)),
            "fire_rows": int((df.fire == 1).sum()),
            "nonfire_rows": int((df.fire == 0).sum()),
            "years": sorted(df.year.unique().tolist()),
            "features": features,
        },
        "temporal": temporal_experiment(df, features),
        "spatial": spatial_experiment(df, features),
        "ablation": ablation(df),
        "protocol": {
            "primary_test": "2025 held out from model selection",
            "validation": "2024",
            "training": "2018-2023",
            "negative_sampling": "1:1 controlled non-fire samples matched to observed fire-cell date/hour universe",
            "target_definition": "FIRMS active-fire detection occurrence at 0.1-degree grid/time cell",
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["ablation"]).to_csv(outdir / "ablation.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
