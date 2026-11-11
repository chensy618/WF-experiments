"""
Measure wall-clock time (and, for FCN3/PyTorch, peak GPU memory) across different
forecast lengths, for the computational-performance section of the rebuttal.

Runs a small number of init times per length (not a full year) so this stays cheap
relative to the ongoing 2010-2015 backfill. Output zarrs are discarded after timing.

Usage
-----
    python compute_cost_by_length.py --model fcn3
    python compute_cost_by_length.py --model graphcast --package-path /cluster/work/.../graphcast_local
"""

import argparse
import shutil
import time
from pathlib import Path

import torch

from config import OUT_ROOT
from forecasting import load_data_source, load_model

LEAD_DAYS = [3, 5, 10, 15]
NSTEPS_BY_DAYS = {d: d * 4 for d in LEAD_DAYS}  # 6-hour steps per day
INIT_TIMES = ["2022-02-01T00:00:00", "2022-06-01T00:00:00", "2022-10-01T00:00:00"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fcn3", "graphcast"], required=True)
    parser.add_argument("--package-path", type=str, default=None)
    args = parser.parse_args()

    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic

    data = load_data_source(None)
    model, model_label = load_model(args.model, args.package_path)

    is_torch = args.model == "fcn3"
    tmp_dir = OUT_ROOT / "tmp_compute_cost_test"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'days':>6}  {'nsteps':>7}  {'init_time':>22}  {'wall_s':>9}  {'peak_mem_GB':>12}")
    rows = []
    for days in LEAD_DAYS:
        nsteps = NSTEPS_BY_DAYS[days]
        for t in INIT_TIMES:
            if is_torch:
                torch.cuda.reset_peak_memory_stats()
            t_zarr = tmp_dir / f"{model_label.lower()}_{days}d_{t.replace(':', '')}.zarr"
            if t_zarr.exists():
                shutil.rmtree(t_zarr)
            io = ZarrBackend(str(t_zarr))

            t0 = time.time()
            deterministic([t], nsteps, model, data, io)
            elapsed = time.time() - t0

            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if is_torch else float("nan")
            shutil.rmtree(t_zarr, ignore_errors=True)

            print(f"{days:>6}  {nsteps:>7}  {t:>22}  {elapsed:>9.1f}  {peak_mem_gb:>12.2f}")
            rows.append({"model": model_label, "days": days, "nsteps": nsteps,
                         "init_time": t, "wall_s": elapsed, "peak_mem_gb": peak_mem_gb})

    shutil.rmtree(tmp_dir, ignore_errors=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = OUT_ROOT / f"compute_cost_by_length_{args.model}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    summary = df.groupby(["days", "nsteps"]).agg(
        mean_wall_s=("wall_s", "mean"), std_wall_s=("wall_s", "std"),
        mean_peak_mem_gb=("peak_mem_gb", "mean"),
    ).reset_index()
    print("\nSummary (mean over 3 init times per length):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
