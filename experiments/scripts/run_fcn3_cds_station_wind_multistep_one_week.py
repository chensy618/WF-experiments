import argparse
from datetime import datetime, timedelta
from pathlib import Path

import earth2studio.run as run
from earth2studio.data import CDS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run FCN3 with CDS ERA5 data for one weekly station-wind chunk. "
            "This script is for multi-step forecasts: +6h, +12h, +18h, +24h."
        )
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date of this weekly chunk, e.g. 2016-01-01.",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of initialization days in this chunk.",
    )

    parser.add_argument(
        "--nsteps",
        type=int,
        default=4,
        help="Number of 6-hour forecast steps. 4 means +6h,+12h,+18h,+24h.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the weekly Zarr output will be saved.",
    )

    return parser.parse_args()


def build_init_times(start_date: str, days: int):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=days - 1)

    init_times = [
        (start_dt + timedelta(days=i)).strftime("%Y-%m-%dT00:00:00")
        for i in range(days)
    ]

    return init_times, start_dt, end_dt


def main():
    args = parse_args()

    init_times, start_dt, end_dt = build_init_times(
        start_date=args.start_date,
        days=args.days,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_start = start_dt.strftime("%Y%m%d")
    safe_end = end_dt.strftime("%Y%m%d")

    output_path = (
        output_dir
        / f"fcn3_station_wind_multistep_{safe_start}_{safe_end}.zarr"
    )

    lead_hours = [6 * i for i in range(1, args.nsteps + 1)]

    print("=" * 80)
    print("Running FCN3 + CDS station wind multi-step weekly forecast")
    print(f"Start date:  {start_dt.strftime('%Y-%m-%d')}")
    print(f"End date:    {end_dt.strftime('%Y-%m-%d')}")
    print(f"Days:        {args.days}")
    print(f"Init times:  {init_times}")
    print(f"N steps:     {args.nsteps}")
    print(f"Lead hours:  {lead_hours}")
    print(f"Output:      {output_path}")
    print("=" * 80)

    if output_path.exists():
        print("Output already exists. Skipping.")
        print(f"Existing output: {output_path}")
        return

    model = FCN3.load_model(FCN3.load_default_package())
    data = CDS()

    io = ZarrBackend(
        file_name=str(output_path),
        chunks={
            "time": 1,
            "lead_time": 1,
            "variable": 1,
        },
        backend_kwargs={
            "overwrite": True,
        },
    )

    run.deterministic(
        init_times,
        args.nsteps,
        model,
        data,
        io,
    )

    print("=" * 80)
    print("Finished FCN3 + CDS station wind multi-step forecast successfully.")
    print(f"Saved output: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()