from __future__ import annotations

import argparse
from pathlib import Path
import cdsapi

DEFAULT_AREA = [37, 68, 6, 98]  # North, West, South, East
YEARS = list(range(2018, 2026))

DAILY_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
]


def valid_days():
    return [f"{d:02d}" for d in range(1, 32)]


def download_daily(client, year: int, out_dir: Path, area: list[float]) -> Path:
    out = out_dir / f"era5_land_daily_mean_{year}.nc.zip"
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP daily {year}: {out} already exists")
        return out

    tmp = out.with_suffix(out.suffix + ".part")
    request = {
        "variable": DAILY_VARIABLES,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": valid_days(),
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": area,
    }
    print(f"START daily statistics {year} -> {out}", flush=True)
    client.retrieve("derived-era5-land-daily-statistics", request, str(tmp))
    tmp.replace(out)
    print(f"DONE daily statistics {year}: {out.stat().st_size:,} bytes", flush=True)
    return out


def download_precip(client, year: int, out_dir: Path, area: list[float]) -> Path:
    out = out_dir / f"era5_land_precip_00utc_{year}.nc"
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP precipitation {year}: {out} already exists")
        return out

    tmp = out.with_suffix(out.suffix + ".part")
    request = {
        "variable": ["total_precipitation"],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": valid_days(),
        "time": ["00:00"],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    print(f"START precipitation {year} -> {out}", flush=True)
    client.retrieve("reanalysis-era5-land", request, str(tmp))
    tmp.replace(out)
    print(f"DONE precipitation {year}: {out.stat().st_size:,} bytes", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Memory-efficient overnight ERA5-Land downloader for Project 1.")
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--output-dir", default="data/raw/era5_land_compact")
    p.add_argument("--area", nargs=4, type=float, default=DEFAULT_AREA, metavar=("N", "W", "S", "E"))
    p.add_argument("--daily-only", action="store_true")
    p.add_argument("--precip-only", action="store_true")
    args = p.parse_args()

    if args.daily_only and args.precip_only:
        raise SystemExit("Use only one of --daily-only or --precip-only.")

    years = list(range(args.start_year, args.end_year + 1))
    if not years:
        raise SystemExit("Invalid year range.")

    daily_dir = Path(args.output_dir) / "daily_mean"
    precip_dir = Path(args.output_dir) / "precip_00utc"
    daily_dir.mkdir(parents=True, exist_ok=True)
    precip_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    print(f"Years: {years[0]}-{years[-1]}")
    print(f"Area: N={args.area[0]}, W={args.area[1]}, S={args.area[2]}, E={args.area[3]}")
    print("Mode: sequential/resumable; one CDS job at a time")

    for year in years:
        if not args.precip_only:
            download_daily(client, year, daily_dir, args.area)
        if not args.daily_only:
            download_precip(client, year, precip_dir, args.area)

    print("ALL REQUESTED ERA5-LAND DOWNLOADS COMPLETE")


if __name__ == "__main__":
    main()
