import argparse
from datetime import datetime, timedelta
from pathlib import Path

import earth2studio.run as run
from earth2studio.data import CDS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import GraphCastOperational


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GraphCastOperational + CDS ERA5 for one week, one daily forecast at a time."
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Week start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of daily forecasts to run from start-date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2012-12-31",
        help="Do not run days after this date. Use YYYY-MM-DD.",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=4,
        help="Number of 6-hour GraphCast steps per daily forecast. 4 steps = 1 day.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where daily Zarr outputs will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    week_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Running GraphCastOperational + CDS ERA5 weekly job")
    print(f"Week start:  {args.start_date}")
    print(f"Days:        {args.days}")
    print(f"End date:    {args.end_date}")
    print(f"N steps/day: {args.nsteps}")
    print(f"Output dir:  {output_dir}")
    print("=" * 80, flush=True)

    print("Loading GraphCastOperational package...", flush=True)
    package = GraphCastOperational.load_default_package()

    print("Loading GraphCastOperational model...", flush=True)
    model = GraphCastOperational.load_model(package)

    print("Initializing CDS ERA5 data source...", flush=True)
    data = CDS(cache=True, verbose=True)

    for day_offset in range(args.days):
        current_day = week_start + timedelta(days=day_offset)

        if current_day > end_date:
            print(
                f"Skipping {current_day.strftime('%Y-%m-%d')} because it is after end-date {args.end_date}",
                flush=True,
            )
            continue

        start_date = current_day.strftime("%Y-%m-%d")
        start_time = f"{start_date}T00:00:00"
        safe_date = start_date.replace("-", "")

        output_path = output_dir / f"graphcast_operational_cds_daily_{safe_date}.zarr"

        if output_path.exists():
            print(f"Output already exists, skipping: {output_path}", flush=True)
            continue

        print("-" * 80)
        print("Running daily GraphCast forecast")
        print(f"Start date:  {start_date}")
        print(f"Start time:  {start_time}")
        print(f"N steps:     {args.nsteps}")
        print(f"Output path: {output_path}")
        print("-" * 80, flush=True)

        io = ZarrBackend(str(output_path))

        run.deterministic(
            [start_time],
            args.nsteps,
            model,
            data,
            io,
        )

        print(f"Finished daily forecast: {start_date}", flush=True)

    print("=" * 80)
    print("Finished weekly GraphCastOperational + CDS ERA5 job")
    print(f"Week start: {args.start_date}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()