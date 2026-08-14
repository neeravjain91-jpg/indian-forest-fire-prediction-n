from __future__ import annotations

import argparse
from pathlib import Path
import time

import cdsapi
import pandas as pd

DATASET = "reanalysis-era5-land-timeseries"
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "volumetric_soil_water_layer_1",
]


def load_events(path: Path, max_events: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"grid_lat", "grid_lon", "acq_date", "hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["grid_lat"] = pd.to_numeric(df["grid_lat"], errors="coerce")
    df["grid_lon"] = pd.to_numeric(df["grid_lon"], errors="coerce")
    df = df.dropna(subset=["grid_lat", "grid_lon", "acq_date", "hour"]).copy()
    df["event_time"] = df["acq_date"] + pd.to_timedelta(df["hour"].astype(int), unit="h")
    return df.head(max_events).copy()


def request_point(client: cdsapi.Client, row: pd.Series, output: Path) -> None:
    event_time = pd.Timestamp(row["event_time"])
    # Seven-day history is sufficient for the planned 24/72/168-hour antecedent features.
    start = (event_time - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = event_time.strftime("%Y-%m-%d")
    request = {
        "variable": VARIABLES,
        "location": {
            "latitude": float(row["grid_lat"]),
            "longitude": float(row["grid_lon"]),
        },
        "date": [f"{start}/{end}"],
        "data_format": "csv",
    }
    result = client.retrieve(DATASET, request)
    result.download(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot point-based ERA5-Land extraction for Project 1.")
    parser.add_argument("--fire-input", required=True)
    parser.add_argument("--output-dir", default="data/raw/era5_land_points")
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--pause", type=float, default=2.0)
    args = parser.parse_args()

    events = load_events(Path(args.fire_input), args.max_events)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    print(f"Pilot events: {len(events)}")
    print(f"Variables: {', '.join(VARIABLES)}")
    print("Window: event time minus 7 days through event date")

    for i, (_, row) in enumerate(events.iterrows(), start=1):
        stamp = pd.Timestamp(row["event_time"]).strftime("%Y%m%d%H")
        lat = float(row["grid_lat"])
        lon = float(row["grid_lon"])
        output = output_dir / f"point_{lat:+07.2f}_{lon:+08.2f}_{stamp}.csv"
        if output.exists() and output.stat().st_size > 0:
            print(f"[{i}/{len(events)}] exists: {output.name}")
            continue
        print(f"[{i}/{len(events)}] requesting lat={lat:.2f}, lon={lon:.2f}, event={row['event_time']}")
        request_point(client, row, output)
        print(f"[{i}/{len(events)}] saved: {output.name} ({output.stat().st_size:,} bytes)")
        if i < len(events):
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
