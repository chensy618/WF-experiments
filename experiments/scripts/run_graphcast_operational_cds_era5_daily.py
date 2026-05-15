import argparse
from datetime import datetime, timedelta
from pathlib import Path

import earth2studio.run as run
from earth2studio.data import CDS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import GraphCastOperational


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GraphCastOperational with CDS ERA5 data for daily forecasts."
    )

    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Daily task index. task-id=1 starts from base-date.",
    )
    parser.add_argument(
        "--base-date",
        type=str,
        default="2009-01-01T00:00:00",
        help="Base start datetime in ISO format.",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=4,
        help="Number of 6-hour forecast steps. 4 steps = 1 day.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/outputs/graphcast_operational_cds_era5_daily",
        help="Output directory for daily Zarr forecast files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    base_time = datetime.fromisoformat(args.base_date)
    start_time = base_time + timedelta(days=args.task_id - 1)

    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    safe_start = start_time.strftime("%Y%m%d_%H")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_path = output_root / f"graphcast_operational_cds_era5_daily_{safe_start}.zarr"

    print("=" * 80)
    print("GraphCastOperational + CDS ERA5 daily forecast")
    print(f"Task ID:      {args.task_id}")
    print(f"Base date:    {args.base_date}")
    print(f"Start time:   {start_str}")
    print(f"N steps:      {args.nsteps}")
    print(f"Output path:  {output_path}")
    print("=" * 80, flush=True)

    print("Loading GraphCastOperational package...", flush=True)
    package = GraphCastOperational.load_default_package()

    print("Loading GraphCastOperational model...", flush=True)
    model = GraphCastOperational.load_model(package)

    print("Initializing CDS ERA5 data source...", flush=True)
    data = CDS(cache=True, verbose=True)

    print("Initializing Zarr output backend...", flush=True)
    io = ZarrBackend(str(output_path))

    print("Running deterministic forecast...", flush=True)
    run.deterministic(
        [start_str],
        args.nsteps,
        model,
        data,
        io,
    )

    print("=" * 80)
    print("Finished successfully")
    print(f"Saved to: {output_path}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()