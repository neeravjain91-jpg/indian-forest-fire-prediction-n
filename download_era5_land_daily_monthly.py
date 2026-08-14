from __future__ import annotations

import argparse
import time
from calendar import monthrange
from pathlib import Path

import cdsapi

AREA = [37.0, 68.0, 6.0, 98.0]  # N, W, S, E
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
]


def days_for(year: int, month: int) -> list[str]:
    return [f"{d:02d}" for d in range(1, monthrange(year, month)[1] + 1)]


def retrieve_with_retry(client, dataset: str, request: dict, target: Path, retries: int = 4) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"SKIP: {target}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            if partial.exists():
                partial.unlink()
            print(f"REQUEST {attempt}/{retries}: {dataset} -> {target.name}", flush=True)
            client.retrieve(dataset, request, str(partial))
            if partial.stat().st_size == 0:
                raise RuntimeError("CDS returned an empty file")
            partial.replace(target)
            print(f"DONE: {target.name} ({target.stat().st_size:,} bytes)", flush=True)
            return
        except Exception as exc:
            print(f"FAILED attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            if attempt < retries:
                delay = min(300, 30 * attempt)
                print(f"Retrying in {delay}s...", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Failed after {retries} attempts: {target}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Resumable monthly ERA5-Land daily-statistics + precipitation downloader for Project 1."
    )
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--output-dir", default="data/raw/era5_land_daily")
    p.add_argument("--area", nargs=4, type=float, default=AREA, metavar=("N", "W", "S", "E"))
    p.add_argument("--start-month", type=int, default=1)
    p.add_argument("--max-months", type=int, default=None)
    args = p.parse_args()

    if not (1 <= args.start_month <= 12):
        raise SystemExit("--start-month must be 1..12")
    if args.start_year > args.end_year:
        raise SystemExit("start-year must be <= end-year")

    jobs = []
    for year in range(args.start_year, args.end_year + 1):
        first_month = args.start_month if year == args.start_year else 1
        for month in range(first_month, 13):
            jobs.append((year, month))
    if args.max_months is not None:
        jobs = jobs[: args.max_months]

    out_root = Path(args.output_dir)
    daily_dir = out_root / "daily_mean"
    precip_dir = out_root / "precip_00utc"
    client = cdsapi.Client()

    print(f"Months: {len(jobs)}", flush=True)
    print(f"Area: N={args.area[0]}, W={args.area[1]}, S={args.area[2]}, E={args.area[3]}", flush=True)
    print("Daily variables:", ", ".join(VARIABLES), flush=True)
    print("One CDS job at a time; completed files are skipped automatically.", flush=True)

    for index, (year, month) in enumerate(jobs, start=1):
        days = days_for(year, month)
        ym = f"{year}_{month:02d}"
        print(f"\n===== {index}/{len(jobs)}: {ym} =====", flush=True)

        daily_request = {
            "variable": VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": args.area,
        }
        daily_target = daily_dir / f"era5_land_daily_mean_{ym}.nc.zip"
        retrieve_with_retry(
            client,
            "derived-era5-land-daily-statistics",
            daily_request,
            daily_target,
        )

        precip_request = {
            "variable": ["total_precipitation"],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": ["00:00"],
            "area": args.area,
            "data_format": "grib",
            "download_format": "unarchived",
        }
        precip_target = precip_dir / f"era5_land_precip_00utc_{ym}.grib"
        retrieve_with_retry(
            client,
            "reanalysis-era5-land",
            precip_request,
            precip_target,
        )

    print("\nALL MONTHS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
