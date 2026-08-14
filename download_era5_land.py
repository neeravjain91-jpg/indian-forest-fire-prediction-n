from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi


VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
    "volumetric_soil_water_layer_1",
]


def request_month(client: cdsapi.Client, year: int, month: int, output: Path, area: list[float]) -> None:
    days = [f"{d:02d}" for d in range(1, 32)]
    times = [f"{h:02d}:00" for h in range(24)]
    request = {
        "variable": VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": times,
        "area": area,  # North, West, South, East
        "data_format": "grib",
        "download_format": "unarchived",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Requesting ERA5-Land: {year}-{month:02d} -> {output}", flush=True)
    client.retrieve("reanalysis-era5-land", request, str(output))
    print(f"Downloaded: {output} ({output.stat().st_size:,} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download bulk ERA5-Land data for Project 1.")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--output-dir", default="data/raw/era5_land")
    parser.add_argument("--area", nargs=4, type=float, default=[38.0, 68.0, 6.0, 98.0], metavar=("N", "W", "S", "E"))
    args = parser.parse_args()

    if not 1 <= args.month <= 12:
        raise SystemExit("--month must be between 1 and 12")

    output = Path(args.output_dir) / f"era5_land_{args.year}_{args.month:02d}.grib"
    if output.exists():
        print(f"Already exists, skipping: {output}")
        return

    client = cdsapi.Client()
    request_month(client, args.year, args.month, output, args.area)


if __name__ == "__main__":
    main()
