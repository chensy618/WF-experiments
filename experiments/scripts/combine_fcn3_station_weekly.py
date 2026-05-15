from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import re

import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine weekly FCN3 station wind Zarr outputs into one final Zarr file."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/cluster/projects/nn8106k/siyan/WF-experiments/outputs/fcn3_station_weekly",
        help="Directory containing weekly FCN3 station Zarr outputs.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cluster/projects/nn8106k/siyan/WF-experiments/outputs/fcn3_station_combined",
        help="Directory for the final combined Zarr output.",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="fcn3_station_wind_2009_2012.zarr",
        help="Name of the final combined Zarr output.",
    )

    parser.add_argument(
        "--start",
        type=str,
        default="2009-01-01 00:00",
        help="Start time to keep in final dataset.",
    )

    parser.add_argument(
        "--end",
        type=str,
        default="2012-12-31 18:00",
        help="End time to keep in final dataset.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite final output if it already exists.",
    )

    return parser.parse_args()


def extract_start_tag(path: Path) -> str:
    """
    Extract start date tag from filename like:
    fcn3_station_wind_20090101_20090107.zarr
    """
    match = re.search(r"fcn3_station_wind_(\d{8})_(\d{8})\.zarr$", path.name)

    if not match:
        raise ValueError(f"Unexpected weekly filename format: {path.name}")

    return match.group(1)


def find_weekly_outputs(input_dir: Path) -> list[Path]:
    paths = sorted(
        input_dir.glob("fcn3_station_wind_*.zarr"),
        key=extract_start_tag,
    )

    return paths


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output_name

    if output_path.exists():
        if args.overwrite:
            import shutil

            print(f"Removing existing output: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                f"Use --overwrite if you want to replace it."
            )

    print("============================================================")
    print("Combining FCN3 weekly station wind outputs")
    print("============================================================")
    print("Input directory:", input_dir)
    print("Output path:", output_path)
    print("Start:", args.start)
    print("End:", args.end)

    paths = find_weekly_outputs(input_dir)

    print("Found weekly outputs:", len(paths))

    if not paths:
        raise FileNotFoundError(f"No weekly Zarr outputs found in: {input_dir}")

    print("First file:", paths[0])
    print("Last file:", paths[-1])

    datasets = []

    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] Opening: {path.name}")
        ds = xr.open_zarr(path)
        datasets.append(ds)

    print("Combining datasets...")
    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    )

    if "time" not in combined.coords:
        raise ValueError("Combined dataset has no 'time' coordinate.")

    print("Sorting by time...")
    combined = combined.sortby("time")

    print("Trimming to requested time range...")
    combined = combined.sel(time=slice(args.start, args.end))

    print("Checking for duplicate time values...")
    time_index = combined.indexes["time"]

    if time_index.has_duplicates:
        print("Duplicate time values detected. Removing duplicates, keeping first occurrence.")
        combined = combined.sel(time=~time_index.duplicated())

    combined.attrs["model"] = "FCN3"
    combined.attrs["input_data"] = "CDS ERA5"
    combined.attrs["chunking"] = "weekly"
    combined.attrs["combined_from"] = str(input_dir)
    combined.attrs["created_at"] = datetime.now().isoformat()
    combined.attrs["start_time"] = args.start
    combined.attrs["end_time"] = args.end
    combined.attrs["description"] = (
        "Combined weekly station-nearest 10 m wind speed outputs from FCN3. "
        "Only station-level extracted values are stored, not full global forecast fields."
    )

    print("Final combined dataset:")
    print(combined)

    print("Saving combined Zarr...")
    combined.to_zarr(output_path, mode="w")

    for ds in datasets:
        ds.close()

    print("============================================================")
    print("Finished successfully")
    print("Saved combined output to:")
    print(output_path)
    print("============================================================")


if __name__ == "__main__":
    main()