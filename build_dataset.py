from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover",
    "soil_moisture_0_to_7cm",
]
DERIVED = ["rain_24h", "rain_72h", "rain_168h", "avg_temp_24h", "avg_humidity_24h", "max_wind_24h"]
FEATURES = BASE + DERIVED
API = "https://archive-api.open-meteo.com/v1/archive"


def request_weather(points: list[tuple[float, float]], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    params = {
        "latitude": ",".join(f"{x:.1f}" for x, _ in points),
        "longitude": ",".join(f"{y:.1f}" for _, y in points),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(BASE),
        "models": "ecmwf_ifs",
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }
    for attempt in range(8):
        try:
            r = requests.get(API, params=params, timeout=180)
            if r.status_code == 429:
                wait = min(600, 60 * (attempt + 1))
                time.sleep(wait)
                continue
            if r.status_code in (502, 503, 504):
                time.sleep(min(240, 30 * (attempt + 1)))
                continue
            r.raise_for_status()
            payload = r.json()
            return payload if isinstance(payload, list) else [payload]
        except requests.RequestException:
            if attempt == 7:
                return []
            time.sleep(min(240, 30 * (attempt + 1)))
    return []


def feature_at(hourly: dict, target: pd.Timestamp) -> dict | None:
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    if len(times) == 0:
        return None
    target = pd.Timestamp(target).tz_localize("UTC") if pd.Timestamp(target).tzinfo is None else pd.Timestamp(target).tz_convert("UTC")
    matches = np.where(times == target)[0]
    if len(matches) == 0:
        return None
    i = int(matches[0])
    out = {c: hourly.get(c, [np.nan] * len(times))[i] for c in BASE}

    def window(col: str, hours: int) -> pd.Series:
        values = pd.to_numeric(pd.Series(hourly.get(col, [])), errors="coerce")
        return values.iloc[max(0, i - hours + 1): i + 1]

    out["rain_24h"] = window("precipitation", 24).sum(min_count=1)
    out["rain_72h"] = window("precipitation", 72).sum(min_count=1)
    out["rain_168h"] = window("precipitation", 168).sum(min_count=1)
    out["avg_temp_24h"] = window("temperature_2m", 24).mean()
    out["avg_humidity_24h"] = window("relative_humidity_2m", 24).mean()
    out["max_wind_24h"] = window("wind_speed_10m", 24).max()
    return out


def sample_nonfire(fire: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = set(zip(fire.grid_lat.round(1), fire.grid_lon.round(1), fire.acq_date.dt.strftime("%Y-%m-%d"), fire.hour))
    dates = fire.acq_date.dt.normalize().drop_duplicates().tolist()
    cells = fire[["grid_lat", "grid_lon"]].drop_duplicates().to_numpy()
    rows = []
    target = len(fire)
    attempts = 0
    while len(rows) < target and attempts < target * 100:
        attempts += 1
        cell = cells[rng.integers(0, len(cells))]
        date = dates[rng.integers(0, len(dates))]
        hour = int(rng.integers(0, 24))
        key = (round(float(cell[0]), 1), round(float(cell[1]), 1), pd.Timestamp(date).strftime("%Y-%m-%d"), hour)
        if key in keys:
            continue
        rows.append({"grid_lat": key[0], "grid_lon": key[1], "acq_date": pd.Timestamp(date), "hour": hour, "fire": 0})
    if len(rows) < target:
        raise RuntimeError("Unable to create the requested number of non-fire samples.")
    return pd.DataFrame(rows)


def build_weather(df: pd.DataFrame, cache_path: Path, batch_size: int = 50, pause: float = 2.0) -> pd.DataFrame:
    cache = {}
    if cache_path.exists():
        old = pd.read_csv(cache_path)
        for _, r in old.iterrows():
            key = (round(float(r.grid_lat), 1), round(float(r.grid_lon), 1), str(r.acq_date), int(r.hour))
            cache[key] = r.to_dict()

    keys = [(round(float(r.grid_lat), 1), round(float(r.grid_lon), 1), pd.Timestamp(r.acq_date).strftime("%Y-%m-%d"), int(r.hour)) for _, r in df.iterrows()]
    missing = list(dict.fromkeys(k for k in keys if k not in cache))
    print(f"Unique weather cells: {len(keys):,}; cached: {len(keys)-len(missing):,}; missing: {len(missing):,}")

    groups: dict[str, list[tuple]] = {}
    for k in missing:
        week = pd.Timestamp(k[2]) - pd.Timedelta(days=pd.Timestamp(k[2]).dayofweek)
        groups.setdefault(week.strftime("%Y-%m-%d"), []).append(k)

    for block_no, (week_key, block_keys) in enumerate(groups.items(), 1):
        start = pd.Timestamp(week_key) - pd.Timedelta(days=7)
        end = pd.Timestamp(week_key) + pd.Timedelta(days=6)
        coords = list(dict.fromkeys((k[0], k[1]) for k in block_keys))
        for i in range(0, len(coords), batch_size):
            batch = coords[i:i + batch_size]
            payloads = request_weather(batch, start, end)
            for (lat, lon), payload in zip(batch, payloads):
                hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
                for k in block_keys:
                    if k[:2] != (lat, lon):
                        continue
                    target = pd.Timestamp(k[2]) + pd.Timedelta(hours=k[3])
                    vals = feature_at(hourly, target)
                    if vals is not None:
                        cache[k] = {"grid_lat": lat, "grid_lon": lon, "acq_date": k[2], "hour": k[3], **vals}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cache.values()).to_csv(cache_path, index=False)
            time.sleep(pause)
        print(f"Weather block {block_no}/{len(groups)} complete; cache={len(cache):,}")

    rows = []
    for _, r in df.iterrows():
        k = (round(float(r.grid_lat), 1), round(float(r.grid_lon), 1), pd.Timestamp(r.acq_date).strftime("%Y-%m-%d"), int(r.hour))
        if k in cache:
            row = r.to_dict()
            row.update({c: cache[k].get(c, np.nan) for c in FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fire-input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default="cache/weather_cache.csv")
    p.add_argument("--max-fire-cells", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fire = pd.read_csv(args.fire_input, parse_dates=["acq_date"])
    if args.max_fire_cells:
        fire = fire.sample(min(args.max_fire_cells, len(fire)), random_state=args.seed).reset_index(drop=True)
    nonfire = sample_nonfire(fire, args.seed)
    combined = pd.concat([fire, nonfire], ignore_index=True)
    out = build_weather(combined, Path(args.cache))
    out = out.dropna(subset=FEATURES + ["fire"])
    out["year"] = pd.to_datetime(out.acq_date).dt.year.astype(int)
    out["month"] = pd.to_datetime(out.acq_date).dt.month.astype(int)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Final weather-matched rows: {len(out):,}")
    print(f"Fire rows: {(out.fire == 1).sum():,}; non-fire rows: {(out.fire == 0).sum():,}")


if __name__ == "__main__":
    main()
