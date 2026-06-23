"""
Download ECMWF HRES operational forecast data via MARS WebAPI.

Downloads surface and pressure-level variables at 00/12 UTC initializations,
steps 0-240 h in 6 h intervals, for the requested years.

Variable set matches the WeatherBench2 HRES zarr (2016-2022-0012-1440x721.zarr)
so that 2024 (and any other years) can be compared consistently.

Output files
------------
    hres_sfc_<year>_<month>.grib   -- 6 surface variables
    hres_pl_<year>_<month>.grib    -- 6 pressure-level variables x 13 levels

Prerequisites
-------------
    pip install ecmwf-api-client
    ~/.ecmwfapirc  with url / key / email (see https://confluence.ecmwf.int/display/WEBAPI/Access+MARS)

Usage
-----
    python download_hres_mars.py --years 2024
    python download_hres_mars.py --years 2016 2018 2022 2024 --output-dir /path/to/dir
"""

import argparse
import calendar
from pathlib import Path

from ecmwfapi import ECMWFService

DEFAULT_OUTPUT_DIR = Path("/cluster/work/projects/nn8106k/siyan/hres_mars")

# GRIB parameter IDs
SURFACE_PARAMS = "165/166/167/151/134/228"
# u10, v10, 2m_t, msl, sp, tp (total precip – accumulated)

PRESSURE_PARAMS = "129/130/131/132/133/135"
# z, t, u, v, q, w (vertical velocity / omega)

PRESSURE_LEVELS = "50/100/150/200/250/300/400/500/600/700/850/925/1000"

STEPS = "/".join(str(s) for s in range(0, 241, 6))  # 0, 6, 12, … 240
TIMES = "00:00:00/12:00:00"
GRID = "0.25/0.25"
AREA = "90/-180/-90/179.75"   # N/W/S/E – explicit global coverage


def _date_range(year: int, month: int) -> str:
    """Return 'YYYYMMDD/to/YYYYMMDD' for the given month."""
    last = calendar.monthrange(year, month)[1]
    return f"{year}{month:02d}01/to/{year}{month:02d}{last}"


def _base_request(year: int, month: int) -> dict:
    return {
        "class":   "od",
        "expver":  "1",
        "stream":  "oper",
        "type":    "fc",
        "date":    _date_range(year, month),
        "time":    TIMES,
        "step":    STEPS,
        "grid":    GRID,
        "area":    AREA,
        "format":  "grib",
    }


def download_surface(
    mars: ECMWFService, year: int, month: int, output_dir: Path
) -> None:
    out = output_dir / f"hres_sfc_{year}_{month:02d}.grib"
    if out.exists():
        print(f"  Skipping {out.name} (already exists)")
        return
    print(f"  Requesting surface {year}-{month:02d} ...")
    req = _base_request(year, month)
    req.update({"levtype": "sfc", "param": SURFACE_PARAMS})
    mars.execute(req, str(out))
    print(f"  Saved -> {out}")


def download_pressure(
    mars: ECMWFService, year: int, month: int, output_dir: Path
) -> None:
    out = output_dir / f"hres_pl_{year}_{month:02d}.grib"
    if out.exists():
        print(f"  Skipping {out.name} (already exists)")
        return
    print(f"  Requesting pressure-level {year}-{month:02d} ...")
    req = _base_request(year, month)
    req.update({
        "levtype":  "pl",
        "levelist": PRESSURE_LEVELS,
        "param":    PRESSURE_PARAMS,
    })
    mars.execute(req, str(out))
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
        "--surface-only", action="store_true",
        help="Download only surface-level variables",
    )
    parser.add_argument(
        "--pressure-only", action="store_true",
        help="Download only pressure-level variables",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mars = ECMWFService("mars")

    for year in args.years:
        print(f"\n{'='*50}")
        print(f"Year {year}")
        print(f"{'='*50}")
        for month in range(1, 13):
            if not args.pressure_only:
                download_surface(mars, year, month, args.output_dir)
            if not args.surface_only:
                download_pressure(mars, year, month, args.output_dir)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
