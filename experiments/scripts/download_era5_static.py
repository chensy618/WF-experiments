"""
Download ERA5 time-invariant terrain fields (land-sea mask, orography, sub-grid
orography parameters) for station-level terrain diagnostics.

These fields do not vary with time, so a single arbitrary timestamp is requested
and the file is tiny (global 0.25 deg, 7 variables, no time series).

Usage
-----
    python download_era5_static.py
    python download_era5_static.py --output-dir /path/to/dir

Requirements
------------
A valid ~/.cdsapirc with CDS credentials (same as download_era5_6h.py):
    url: https://cds.climate.copernicus.eu/api
    key: <your-api-key>
"""

import argparse
from pathlib import Path

import cdsapi


DEFAULT_OUTPUT_DIR = Path("/cluster/work/projects/nn8106k/siyan/era5_6h")

# Time-invariant fields. `land_sea_mask` and `geopotential` (surface orography,
# geopotential / 9.80665 = height in m) directly test whether a station is seen
# as land vs ocean and how tall the model thinks the terrain is there. The four
# sub-grid orography parameters feed the IFS orographic drag scheme and are the
# most direct proxy for "how complex does the model think this grid cell's
# terrain is" (Køltzow et al.-style roughness/drag argument raised in review).
STATIC_VARS = [
    "land_sea_mask",
    "geopotential",
    "standard_deviation_of_orography",
    "anisotropy_of_sub_gridscale_orography",
    "slope_of_sub_gridscale_orography",
    "angle_of_sub_gridscale_orography",
]


def download_static(c: cdsapi.Client, output_dir: Path) -> None:
    out = output_dir / "era5_static.nc"
    if out.exists():
        print(f"Skipping {out.name} (already exists)")
        return
    print("Requesting ERA5 static terrain fields ...")
    req = {
        "product_type": "reanalysis",
        "variable": STATIC_VARS,
        # Arbitrary single timestamp - these fields are time-invariant.
        "year": "2020",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "format": "netcdf",
        "grid": "0.25/0.25",
    }
    c.retrieve("reanalysis-era5-single-levels", req, str(out))
    print(f"Saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    download_static(c, args.output_dir)


if __name__ == "__main__":
    main()
