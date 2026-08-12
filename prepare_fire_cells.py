from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def round_grid(x: pd.Series) -> pd.Series:
    return (pd.to_numeric(x, errors="coerce") / 0.1).round() * 0.1


def load_firms(path: Path) -> pd.DataFrame:
    usecols = [
        "latitude", "longitude", "acq_date", "acq_time", "confidence", "frp",
        "satellite", "instrument", "daynight", "type"
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["acq_time"] = pd.to_numeric(df["acq_time"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")

    # FIRMS confidence is categorical in this export (for example l/n/h).
    # Convert it to an ordinal score only for aggregation; retain no fire-label information.
    confidence_raw = df["confidence"].astype("string").str.strip().str.lower()
    confidence_map = {"l": 0.25, "n": 0.50, "h": 1.00}
    df["confidence_score"] = confidence_raw.map(confidence_map)
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "acq_date", "acq_time"])
    # Approximate India bounding box; final studies should document this geographic filter.
    df = df[(df.latitude >= 6) & (df.latitude <= 37) & (df.longitude >= 68) & (df.longitude <= 98)]
    df["hour"] = (df["acq_time"] // 100).clip(0, 23).astype(int)
    df["grid_lat"] = round_grid(df["latitude"])
    df["grid_lon"] = round_grid(df["longitude"])
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["grid_lat", "grid_lon", "acq_date", "hour"], as_index=False)
        .agg(
            fire_detections=("latitude", "size"),
            mean_confidence=("confidence_score", "mean"),
            max_frp=("frp", "max"),
        )
    )
    grouped["fire"] = 1
    grouped["year"] = grouped["acq_date"].dt.year.astype(int)
    grouped["month"] = grouped["acq_date"].dt.month.astype(int)
    return grouped


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    years = sorted(df.year.unique())
    rng = np.random.default_rng(seed)
    picks = []
    base, remainder = divmod(n, len(years))
    for i, year in enumerate(years):
        part = df[df.year == year]
        take = min(len(part), base + (1 if i < remainder else 0))
        if take:
            picks.append(part.sample(take, random_state=seed + int(year)))
    out = pd.concat(picks, ignore_index=True)
    if len(out) < n:
        remaining = df.drop(out.index, errors="ignore")
        extra = remaining.sample(n - len(out), random_state=seed)
        out = pd.concat([out, extra], ignore_index=True)
    return out.sample(frac=1, random_state=int(rng.integers(1, 2**31 - 1))).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-fire-cells", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = load_firms(Path(args.input))
    cells = aggregate(raw)
    selected = stratified_sample(cells, args.max_fire_cells, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    print(f"Raw FIRMS detections: {len(raw):,}")
    print(f"Unique 0.1° fire cells: {len(cells):,}")
    print(f"Selected fire cells: {len(selected):,}")
    print(f"Year range: {selected.year.min()}–{selected.year.max()}")


if __name__ == "__main__":
    main()
