import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import earth2studio.run as run
from earth2studio.data import CDS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import GraphCastOperational


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run GraphCastOperational + CDS ERA5 for one weekly output. "
            "Each weekly output contains multiple daily initial times."
        )
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
        help="Number of daily initial times in this weekly output.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default="2012-12-31",
        help="Do not run initial times after this date. Use YYYY-MM-DD.",
    )

    parser.add_argument(
        "--nsteps",
        type=int,
        default=4,
        help="Number of 6-hour GraphCast steps per initial time. 4 steps = 1 day.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where weekly Zarr outputs will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    week_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_times = []

    for day_offset in range(args.days):
        current_day = week_start + timedelta(days=day_offset)

        if current_day > end_date:
            print(
                f"Skipping {current_day.strftime('%Y-%m-%d')} because it is after end-date {args.end_date}",
                flush=True,
            )
            continue

        start_times.append(current_day.strftime("%Y-%m-%dT00:00:00"))

    if not start_times:
        print("No valid start times for this weekly task. Nothing to run.", flush=True)
        return

    actual_week_start = datetime.fromisoformat(start_times[0])
    actual_week_end = datetime.fromisoformat(start_times[-1])

    safe_start = actual_week_start.strftime("%Y%m%d")
    safe_end = actual_week_end.strftime("%Y%m%d")

    output_path = output_dir / f"graphcast_operational_cds_weekly_{safe_start}_{safe_end}.zarr"
    tmp_output_path = output_dir / f".tmp_graphcast_operational_cds_weekly_{safe_start}_{safe_end}.zarr"

    success_marker = output_path / "_SUCCESS"

    print("=" * 80)
    print("Running GraphCastOperational + CDS ERA5 weekly forecast")
    print(f"Requested week start: {args.start_date}")
    print(f"Actual week start:    {actual_week_start.strftime('%Y-%m-%d')}")
    print(f"Actual week end:      {actual_week_end.strftime('%Y-%m-%d')}")
    print(f"Days requested:       {args.days}")
    print(f"Valid initial times:  {len(start_times)}")
    print(f"End date:             {args.end_date}")
    print(f"N steps:              {args.nsteps}")
    print(f"Output path:          {output_path}")
    print("=" * 80, flush=True)

    if output_path.exists() and success_marker.exists():
        print(f"Weekly output already exists and is marked complete. Skipping: {output_path}", flush=True)
        return

    if output_path.exists() and not success_marker.exists():
        print(f"Output exists but has no _SUCCESS marker. Removing incomplete output: {output_path}", flush=True)
        shutil.rmtree(output_path)

    if tmp_output_path.exists():
        print(f"Removing old temporary output: {tmp_output_path}", flush=True)
        shutil.rmtree(tmp_output_path)

    print("Weekly forecast initial times:")
    for t in start_times:
        print(f"  {t}")
    print("=" * 80, flush=True)

    print("Loading GraphCastOperational package...", flush=True)
    package = GraphCastOperational.load_default_package()

    print("Loading GraphCastOperational model...", flush=True)
    model = GraphCastOperational.load_model(package)

    print("Initializing CDS ERA5 data source...", flush=True)
    data = CDS(cache=True, verbose=True)

    print("Initializing temporary Zarr output backend...", flush=True)
    io = ZarrBackend(str(tmp_output_path))

    print("Running deterministic weekly forecast...", flush=True)

    run.deterministic(
        start_times,
        args.nsteps,
        model,
        data,
        io,
    )

    print("Forecast finished. Writing success marker...", flush=True)
    success_file = tmp_output_path / "_SUCCESS"
    success_file.write_text(
        (
            "GraphCastOperational weekly forecast completed successfully.\n"
            f"week_start={actual_week_start.strftime('%Y-%m-%d')}\n"
            f"week_end={actual_week_end.strftime('%Y-%m-%d')}\n"
            f"n_initial_times={len(start_times)}\n"
            f"nsteps={args.nsteps}\n"
        )
    )

    print("Renaming temporary output to final output...", flush=True)
    tmp_output_path.rename(output_path)

    print("=" * 80)
    print("Finished GraphCastOperational + CDS ERA5 weekly forecast")
    print(f"Saved to: {output_path}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()