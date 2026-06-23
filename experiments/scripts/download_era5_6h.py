"""
Download ERA5 6-hourly reanalysis data for FCN3 and GraphCast initialization.

Downloads all 72 FCN3/GraphCast input variables at 00/06/12/18 UTC for each
requested year and saves two NetCDF files per year:

    era5_single_<year>.nc   -- 7 surface variables
    era5_pressure_<year>.nc -- 5 variables x 13 pressure levels

Usage
-----
    python download_era5_6h.py --years 2016 2018 2022 2024
    python download_era5_6h.py --years 2022 --output-dir /path/to/dir

Requirements
------------
A valid ~/.cdsapirc with CDS credentials:
    url: https://cds.climate.copernicus.eu/api
    key: <your-api-key>

Notes
-----
- GraphCast requires two consecutive 6h snapshots per init (t-6h and t0).
  For Jan 1 00:00 initializations, the Dec 31 18:00 snapshot from the
  previous year is needed. Use --prev-dec31 to download it automatically.
- CDS processes requests asynchronously; each request may queue for hours.
- Existing files are skipped, so the script is safe to re-run after failures.
"""

import argparse
from pathlib import Path

import cdsapi


DEFAULT_OUTPUT_DIR = Path("/cluster/work/projects/nn8106k/siyan/era5_6h")

SINGLE_LEVEL_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
]

PRESSURE_LEVEL_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
]

PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250", "300",
    "400", "500", "600", "700", "850", "925", "1000",
]

TIMES = ["00:00", "06:00", "12:00", "18:00"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def _base_request(year: int, months: list[str]) -> dict:
    return {
        "product_type": "reanalysis",
        "year": str(year),
        "month": months,
        "day": DAYS,
        "time": TIMES,
        "format": "netcdf",
        "grid": "0.25/0.25",
    }


def download_single_level(
    c: cdsapi.Client, year: int, months: list[str], output_dir: Path, tag: str = ""
) -> None:
    suffix = f"_{tag}" if tag else ""
    out = output_dir / f"era5_single_{year}{suffix}.nc"
    if out.exists():
        print(f"  Skipping {out.name} (already exists)")
        return
    print(f"  Requesting single-level {year}{suffix} ...")
    req = _base_request(year, months)
    req["variable"] = SINGLE_LEVEL_VARS
    c.retrieve("reanalysis-era5-single-levels", req, str(out))
    print(f"  Saved -> {out}")


def download_pressure_levels(
    c: cdsapi.Client, year: int, months: list[str], output_dir: Path, tag: str = ""
) -> None:
    # Split by month to stay under the CDS per-request field limit (~95k fields/year
    # exceeds the limit; ~8k fields/month is well within it).
    for month in months:
        suffix = f"_{tag}" if tag else ""
        out = output_dir / f"era5_pressure_{year}_{month}{suffix}.nc"
        if out.exists():
            print(f"  Skipping {out.name} (already exists)")
            continue
        print(f"  Requesting pressure-level {year}-{month} ...")
        req = _base_request(year, [month])
        req["variable"] = PRESSURE_LEVEL_VARS
        req["pressure_level"] = PRESSURE_LEVELS
        c.retrieve("reanalysis-era5-pressure-levels", req, str(out))
        print(f"  Saved -> {out}")


def download_prev_dec31(c: cdsapi.Client, year: int, output_dir: Path) -> None:
    """Download Dec 31 18:00 of the previous year for GraphCast Jan 1 init."""
    prev_year = year - 1
    for name, extra in [
        ("single", {"variable": SINGLE_LEVEL_VARS}),
        ("pressure", {"variable": PRESSURE_LEVEL_VARS, "pressure_level": PRESSURE_LEVELS}),
    ]:
        out = output_dir / f"era5_{name}_{prev_year}_dec31.nc"
        if out.exists():
            print(f"  Skipping {out.name} (already exists)")
            continue
        dataset = (
            "reanalysis-era5-single-levels"
            if name == "single"
            else "reanalysis-era5-pressure-levels"
        )
        print(f"  Requesting {name}-level {prev_year} Dec 31 (GraphCast buffer) ...")
        req = {
            "product_type": "reanalysis",
            "year": str(prev_year),
            "month": "12",
            "day": "31",
            "time": "18:00",
            "format": "netcdf",
            "grid": "0.25/0.25",
            **extra,
        }
        c.retrieve(dataset, req, str(out))
        print(f"  Saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2016, 2018, 2022, 2024],
        help="Years to download (default: 2016 2018 2022 2024)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--single-only", action="store_true",
        help="Download only single-level variables",
    )
    parser.add_argument(
        "--pressure-only", action="store_true",
        help="Download only pressure-level variables",
    )
    parser.add_argument(
        "--prev-dec31", action="store_true",
        help="Also download Dec 31 18:00 of the previous year (needed for GraphCast Jan 1 init)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()

    for year in args.years:
        print(f"\n{'='*50}")
        print(f"Year {year}")
        print(f"{'='*50}")

        if args.prev_dec31:
            download_prev_dec31(c, year, args.output_dir)

        if not args.pressure_only:
            download_single_level(c, year, MONTHS, args.output_dir)
        if not args.single_only:
            download_pressure_levels(c, year, MONTHS, args.output_dir)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
