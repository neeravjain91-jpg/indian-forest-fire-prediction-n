from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("india_forest_fire_dataset.csv")
OUT = Path("research/results")
MODEL_OUT = Path("models/research_model.joblib")
BASE = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover", "soil_moisture_0_to_7cm"]
LAG = ["rain_24h", "rain_72h", "rain_168h", "avg_temp_24h", "avg_humidity_24h", "max_wind_24h"]
FEATURES = BASE + LAG

def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def make_models():
    return {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "random_forest": RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Build the weather-matched dataset first.")
    df = pd.read_csv(DATA_PATH)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=FEATURES + ["fire", "acq_date"])
    df["fire"] = df["fire"].astype(int)
    df["year"] = df["acq_date"].dt.year
    train, valid, test = df[df.year <= 2023], df[df.year == 2024], df[df.year == 2025]
    if train.empty or valid.empty or test.empty:
        raise RuntimeError("Temporal split requires observations in 2018–2023, 2024 and 2025.")
    results, best = [], None
    for name, model in make_models().items():
        model.fit(train[FEATURES], train.fire)
        pv = model.predict_proba(valid[FEATURES])[:, 1]
        mv = metrics(valid.fire.to_numpy(), pv); mv.update({"model": name, "split": "2024_validation"}); results.append(mv)
        if best is None or mv["pr_auc"] > best[0]: best = (mv["pr_auc"], name, model)
    pt = best[2].predict_proba(test[FEATURES])[:, 1]
    final = metrics(test.fire.to_numpy(), pt); final.update({"model": best[1], "split": "2025_final_test"}); results.append(final)
    OUT.mkdir(parents=True, exist_ok=True); MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT / "temporal_results.csv", index=False)
    joblib.dump(best[2], MODEL_OUT)
    (OUT / "experiment_summary.json").write_text(json.dumps({"train_years": "2018-2023", "validation_year": 2024, "test_year": 2025, "selected_model": best[1], "selection_metric": "PR-AUC on 2024 validation", "final_test": final}, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))

if __name__ == "__main__": main()
