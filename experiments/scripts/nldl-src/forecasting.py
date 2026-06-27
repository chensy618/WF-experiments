"""
Steps 1–3: load data source, load model weights, run forecasts.
Uses the earth2studio framework throughout.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from config import LEAD_H, NSTEPS, OUT_ROOT, STATIONS


# =============================================================================
# Step 1  Data source
# =============================================================================

def load_data_source(era5_dir: str | None = None):
    """Return a LocalERA5 data source reading from pre-downloaded NetCDF files.

    Parameters
    ----------
    era5_dir : str or None
        Override the ERA5 data directory (default: config._WORK / "era5_6h").
    """
    from local_era5 import ERA5_DIR, LocalERA5
    src_dir = era5_dir or ERA5_DIR
    print(f"[Step 1] Loading local ERA5 data source from: {src_dir}")
    return LocalERA5(era5_dir=src_dir)


# =============================================================================
# Step 2  Model weights
# =============================================================================

def load_model(model_name: str, package_path: str | None = None):
    """Load FCN3 or GraphCastOperational weights via earth2studio.

    Parameters
    ----------
    model_name   : "fcn3" or "graphcast"
    package_path : local path to pre-downloaded weights (skips GCS download)

    Returns
    -------
    model        : earth2studio prognostic model instance
    model_label  : canonical label string ("FCN3" or "GraphCast")
    """
    if model_name == "fcn3":
        from earth2studio.models.px import FCN3
        print("[Step 2] Loading FCN3 weights ...")
        pkg = FCN3.load_default_package() if package_path is None else package_path
        return FCN3.load_model(pkg), "FCN3"

    if model_name == "graphcast":
        from earth2studio.models.auto import Package
        from earth2studio.models.px import GraphCastOperational
        print("[Step 2] Loading GraphCastOperational weights ...")
        pkg = Package(package_path) if package_path else GraphCastOperational.load_default_package()
        return GraphCastOperational.load_model(pkg), "GraphCast"

    raise ValueError(f"Unknown model '{model_name}'. Choose 'fcn3' or 'graphcast'.")


# =============================================================================
# Step 3  Run forecasts
# =============================================================================

def _weekly_ranges(year: int) -> list[tuple[datetime, datetime]]:
    start, end = datetime(year, 1, 1), datetime(year, 12, 31)
    ranges, cur = [], start
    while cur <= end:
        w_end = min(cur + timedelta(days=6), end)
        ranges.append((cur, w_end))
        cur = w_end + timedelta(days=1)
    return ranges


def _week_tag(start: datetime, end: datetime) -> str:
    return f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"


def _build_init_times(model_label: str, week_start: datetime, week_end: datetime) -> list[str]:
    """Return ISO-format init time strings for one week."""
    days = (week_end - week_start).days + 1
    if model_label == "GraphCast":
        # 6-hourly: 00, 06, 12, 18 UTC every day
        return [
            t.strftime("%Y-%m-%dT%H:%M:%S")
            for d in range(days)
            for t in pd.date_range(
                (week_start + timedelta(days=d)).strftime("%Y-%m-%d 00:00"),
                (week_start + timedelta(days=d)).strftime("%Y-%m-%d 18:00"),
                freq="6h",
            )
        ]
    # FCN3: daily 00 UTC
    return [
        (week_start + timedelta(days=d)).strftime("%Y-%m-%dT00:00:00")
        for d in range(days)
    ]


def _extract_station_wind(raw_zarr_path: Path, model_label: str) -> xr.Dataset:
    """Extract nearest-gridpoint 10 m wind speed for all stations."""
    g = zarr.open_group(str(raw_zarr_path), mode="r")

    lats = g["lat"][:]
    lons = g["lon"][:]

    time_raw   = g["time"][:]
    init_times = (
        pd.DatetimeIndex(time_raw)
        if np.issubdtype(time_raw.dtype, np.datetime64)
        else pd.DatetimeIndex([pd.Timestamp(int(t), unit="ns") for t in time_raw])
    )

    lt_raw    = g["lead_time"][:]
    lt_hours  = (
        (lt_raw / np.timedelta64(1, "h")).astype(float)
        if np.issubdtype(lt_raw.dtype, np.timedelta64)
        else lt_raw.astype(float)
    )

    wind_speed = np.sqrt(g["u10m"][:] ** 2 + g["v10m"][:] ** 2).astype(np.float32)

    das = []
    for st in STATIONS:
        lat_i = int(np.argmin(np.abs(lats - st["lat"])))
        lon_i = int(np.argmin(np.abs(lons - (st["lon"] % 360))))
        da = xr.DataArray(
            wind_speed[:, :, lat_i, lon_i],
            dims=["time", "lead_time"],
            coords={
                "time":      init_times,
                "lead_time": lt_hours.astype("timedelta64[h]"),
            },
        ).expand_dims(station=[st["id"]])
        das.append(da)

    ds = xr.concat(das, dim="station").to_dataset(name="wind_speed_10m")
    ds.attrs["model"] = model_label
    return ds


def run_forecasts(
    model,
    model_label: str,
    data,
    year: int,
    nsteps: int = NSTEPS,
    overwrite: bool = False,
) -> Path:
    """Run weekly forecast chunks for a full year using earth2studio.deterministic.

    Saves compact station zarrs to:
        OUT_ROOT / forecasts / {model_label} / {year} / {model}_{week_tag}.zarr

    Returns the directory containing the weekly zarrs.
    """
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic

    fc_dir  = OUT_ROOT / "forecasts" / model_label / str(year)
    raw_dir = fc_dir / "raw"
    fc_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    lead_hours = [6 * i for i in range(1, nsteps + 1)]
    print(f"\n[Step 3] {model_label} forecasts — year {year}")
    print(f"  Output  : {fc_dir}")
    print(f"  Leads   : {lead_hours} h")

    for week_start, week_end in _weekly_ranges(year):
        tag      = _week_tag(week_start, week_end)
        out_zarr = fc_dir / f"{model_label.lower()}_{tag}.zarr"

        if out_zarr.exists() and not overwrite:
            print(f"  {tag}: exists, skipping.")
            continue

        init_times = _build_init_times(model_label, week_start, week_end)

        print(f"  {tag}: {len(init_times)} inits ...", end=" ", flush=True)
        stn_datasets = []
        for i, t in enumerate(init_times):
            t_zarr = raw_dir / f"raw_{tag}_{i:03d}.zarr"
            if t_zarr.exists():
                shutil.rmtree(t_zarr)
            io = ZarrBackend(str(t_zarr))
            deterministic([t], nsteps, model, data, io)
            stn_datasets.append(_extract_station_wind(t_zarr, model_label))
            shutil.rmtree(t_zarr)

        stn_ds = xr.concat(stn_datasets, dim="time")
        stn_ds.to_zarr(out_zarr, mode="w")
        print(f"done ({len(stn_ds.time)} inits saved)")

    print(f"  Forecasts complete → {fc_dir}")
    return fc_dir
