from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import argparse
import subprocess
import sys
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--nsteps", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cluster/projects/nn8106k/siyan/WF-experiments/outputs/fcn3_station_weekly",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=args.days - 1)

    start = start_date.strftime("%Y-%m-%d 00:00")
    end = end_date.strftime("%Y-%m-%d 18:00")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_tag = start_date.strftime("%Y%m%d")
    end_tag = end_date.strftime("%Y%m%d")
    fixed_output = output_dir / f"fcn3_station_wind_{start_tag}_{end_tag}.zarr"

    if fixed_output.exists():
        if args.overwrite:
            print(f"Removing existing output: {fixed_output}")
            shutil.rmtree(fixed_output)
        else:
            print(f"Output already exists, skipping: {fixed_output}")
            return

    print("Weekly FCN3/CDS run")
    print("Start:", start)
    print("End:", end)
    print("Fixed output:", fixed_output)

    before = set(output_dir.glob("fcn3_cds_era5_station_wind_*.zarr"))

    cmd = [
        sys.executable,
        "experiments/scripts/run_fcn3_cds_era5_station_wind_2009_2012.py",
        "--start",
        start,
        "--end",
        end,
        "--nsteps",
        str(args.nsteps),
        "--output-dir",
        str(output_dir),
    ]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    after = set(output_dir.glob("fcn3_cds_era5_station_wind_*.zarr"))
    new_outputs = sorted(after - before, key=lambda p: p.stat().st_mtime)

    if not new_outputs:
        raise RuntimeError("No new timestamped FCN3 station output was created.")

    latest_output = new_outputs[-1]

    print("Renaming:")
    print(latest_output)
    print("to:")
    print(fixed_output)

    latest_output.rename(fixed_output)

    print("Finished successfully.")
    print("Saved:", fixed_output)


if __name__ == "__main__":
    main()