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


def _build_init_times(
    model_label: str, week_start: datetime, week_end: datetime, daily_only: bool = False
) -> list[str]:
    """Return ISO-format init time strings for one week."""
    days = (week_end - week_start).days + 1
    if model_label == "GraphCast" and not daily_only:
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
    # Daily 00 UTC: FCN3 always, or GraphCast when daily_only=True
    return [
        (week_start + timedelta(days=d)).strftime("%Y-%m-%dT00:00:00")
        for d in range(days)
    ]


def _read_raw_wind(raw_zarr_path: Path) -> tuple:
    """Shared raw-zarr readout for both extraction methods below.

    Returns (lats, lons, init_times, lt_hours, wind_speed, u10m, v10m) where
    wind_speed/u10m/v10m each have shape (time, lead_time, lat, lon).
    """
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

    u10m = g["u10m"][:].astype(np.float32)
    v10m = g["v10m"][:].astype(np.float32)
    wind_speed = np.sqrt(u10m ** 2 + v10m ** 2).astype(np.float32)
    return lats, lons, init_times, lt_hours, wind_speed, u10m, v10m


def _extract_station_wind_nearest(
    raw_zarr_path: Path, model_label: str, save_uv: bool = False
) -> xr.Dataset:
    """Extract nearest-gridpoint 10 m wind speed for all stations.

    This is the original extraction method used for every FCN3/GraphCast run
    to date (2016-2022, `_72h` included) — kept unchanged so existing forecast
    zarrs remain reproducible from this function.

    save_uv : if True, also adds `u10m`/`v10m` data variables alongside
        `wind_speed_10m` (which is always present, unchanged). Callers that
        want this must also route to a separate output directory — see
        `run_forecasts`'s `_uv` dir-suffix — so it never overwrites existing
        scalar-only forecast zarrs.
    """
    lats, lons, init_times, lt_hours, wind_speed, u10m, v10m = _read_raw_wind(raw_zarr_path)
    coords = {"time": init_times, "lead_time": lt_hours.astype("timedelta64[h]")}

    das, u_das, v_das = [], [], []
    for st in STATIONS:
        lat_i = int(np.argmin(np.abs(lats - st["lat"])))
        lon_i = int(np.argmin(np.abs(lons - (st["lon"] % 360))))
        das.append(
            xr.DataArray(wind_speed[:, :, lat_i, lon_i], dims=["time", "lead_time"], coords=coords)
            .expand_dims(station=[st["id"]])
        )
        if save_uv:
            u_das.append(
                xr.DataArray(u10m[:, :, lat_i, lon_i], dims=["time", "lead_time"], coords=coords)
                .expand_dims(station=[st["id"]])
            )
            v_das.append(
                xr.DataArray(v10m[:, :, lat_i, lon_i], dims=["time", "lead_time"], coords=coords)
                .expand_dims(station=[st["id"]])
            )

    ds = xr.concat(das, dim="station").to_dataset(name="wind_speed_10m")
    if save_uv:
        ds["u10m"] = xr.concat(u_das, dim="station")
        ds["v10m"] = xr.concat(v_das, dim="station")
    ds.attrs["model"] = model_label
    ds.attrs["extraction_method"] = "nearest"
    return ds


def _extract_station_wind_interp(
    raw_zarr_path: Path, model_label: str, save_uv: bool = False
) -> xr.Dataset:
    """Extract 10 m wind speed for all stations via bilinear interpolation.

    Spatial matching matches stationbench (https://github.com/juaAI/stationbench,
    `interpolate_to_stations` in stationbench/calculate_metrics.py): xarray's
    `.interp(..., method="linear")` to each station's exact coordinates, rather
    than snapping to the nearest 0.25° grid cell.

    save_uv : if True, also adds `u10m`/`v10m` data variables (interpolated the
        same way as `wind_speed_10m`, which is always present, unchanged).
    """
    lats, lons, init_times, lt_hours, wind_speed, u10m, v10m = _read_raw_wind(raw_zarr_path)
    coords = {
        "time":      init_times,
        "lead_time": lt_hours.astype("timedelta64[h]"),
        "lat":       lats,
        "lon":       lons,
    }
    da = xr.DataArray(wind_speed, dims=["time", "lead_time", "lat", "lon"], coords=coords)

    station_ids = [st["id"] for st in STATIONS]
    station_lat = xr.DataArray([st["lat"] for st in STATIONS], dims="station", coords={"station": station_ids})
    station_lon = xr.DataArray([st["lon"] % 360 for st in STATIONS], dims="station", coords={"station": station_ids})

    def _interp(arr: np.ndarray) -> xr.DataArray:
        da_arr = xr.DataArray(arr, dims=["time", "lead_time", "lat", "lon"], coords=coords)
        # xr.interp() with vectorized indexers appends the new "station" dim at
        # the end (time, lead_time, station), unlike the nearest-neighbor
        # extraction's (station, time, lead_time). Transpose so both extraction
        # methods produce identically-shaped output — load_fcn3()/load_graphcast()
        # index positionally and assume the nearest-neighbor layout.
        return da_arr.interp(lat=station_lat, lon=station_lon, method="linear").transpose(
            "station", "time", "lead_time"
        )

    ds = _interp(wind_speed).to_dataset(name="wind_speed_10m")
    if save_uv:
        ds["u10m"] = _interp(u10m)
        ds["v10m"] = _interp(v10m)
    ds.attrs["model"] = model_label
    ds.attrs["extraction_method"] = "interp"
    return ds


_EXTRACTION_FUNCS = {
    "nearest": _extract_station_wind_nearest,
    "interp":  _extract_station_wind_interp,
}


def run_forecasts(
    model,
    model_label: str,
    data,
    year: int,
    nsteps: int = NSTEPS,
    overwrite: bool = False,
    daily_only: bool = False,
    out_tag: str = "",
    extraction: str = "nearest",
    save_uv: bool = False,
) -> Path:
    """Run weekly forecast chunks for a full year using earth2studio.deterministic.

    Saves compact station zarrs to:
        OUT_ROOT / forecasts / {model_label}{out_tag}{dir_suffix} / {year} / {model}_{week_tag}.zarr

    where `dir_suffix` is "" for `extraction="nearest"` (unchanged from every
    prior run, so existing nearest-neighbor forecasts are never touched),
    "_interp" for `extraction="interp"`, and additionally suffixed with "_uv"
    when `save_uv=True` — so runs that also save u10m/v10m always land in
    their own directory and never overwrite (or get skipped-over-by) an
    existing scalar-only forecast run.

    Parameters
    ----------
    daily_only : force 00 UTC-only init (overrides GraphCast's default 6-hourly
        cadence; FCN3 is always daily regardless of this flag).
    out_tag    : suffix appended to the model's forecast directory, e.g. "_72h",
        to keep a longer-horizon run from overwriting the default one.
    extraction : "nearest" (default, original behavior) or "interp" — spatial
        matching method used to pull station values out of the raw global-grid
        forecast. "interp" uses bilinear interpolation to each station's exact
        coordinates, matching stationbench's approach
        (https://github.com/juaAI/stationbench).
    save_uv    : if True, also extract and save `u10m`/`v10m` as additional
        data variables alongside `wind_speed_10m` (which is always saved,
        unchanged). Adds a "_uv" suffix to the output directory.

    Returns the directory containing the weekly zarrs.
    """
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic

    if extraction not in _EXTRACTION_FUNCS:
        raise ValueError(f"Unknown extraction '{extraction}'. Choose from {list(_EXTRACTION_FUNCS)}.")
    extract_fn = _EXTRACTION_FUNCS[extraction]
    dir_suffix = "" if extraction == "nearest" else f"_{extraction}"
    if save_uv:
        dir_suffix += "_uv"

    fc_dir  = OUT_ROOT / "forecasts" / f"{model_label}{out_tag}{dir_suffix}" / str(year)
    raw_dir = fc_dir / "raw"
    fc_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    lead_hours = [6 * i for i in range(1, nsteps + 1)]
    print(f"\n[Step 3] {model_label} forecasts — year {year}")
    print(f"  Output  : {fc_dir}")
    print(f"  Leads   : {lead_hours} h")
    print(f"  Daily-only init: {daily_only}")
    print(f"  Extraction: {extraction}")
    print(f"  Save u10m/v10m: {save_uv}")

    for week_start, week_end in _weekly_ranges(year):
        tag      = _week_tag(week_start, week_end)
        out_zarr = fc_dir / f"{model_label.lower()}_{tag}.zarr"

        if out_zarr.exists() and not overwrite:
            print(f"  {tag}: exists, skipping.")
            continue

        init_times = _build_init_times(model_label, week_start, week_end, daily_only=daily_only)

        print(f"  {tag}: {len(init_times)} inits ...", end=" ", flush=True)
        stn_datasets = []
        for i, t in enumerate(init_times):
            t_zarr = raw_dir / f"raw_{tag}_{i:03d}.zarr"
            if t_zarr.exists():
                shutil.rmtree(t_zarr)
            io = ZarrBackend(str(t_zarr))
            deterministic([t], nsteps, model, data, io)
            stn_datasets.append(extract_fn(t_zarr, model_label, save_uv=save_uv))
            shutil.rmtree(t_zarr)

        stn_ds = xr.concat(stn_datasets, dim="time")
        stn_ds.to_zarr(out_zarr, mode="w")
        print(f"done ({len(stn_ds.time)} inits saved)")

    print(f"  Forecasts complete → {fc_dir}")
    return fc_dir
