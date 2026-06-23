"""
LocalERA5 — earth2studio-compatible data source backed by local NetCDF files.

File layout (all under ERA5_DIR):
    era5_single_{year}.nc            — all 12 months, single-level vars
    era5_single_{year}_dec31.nc      — Dec 31 18:00 of {year}, GraphCast buffer
    era5_pressure_{year}_{MM}.nc     — one file per month, pressure-level vars
    era5_pressure_{year}_dec31.nc    — Dec 31 18:00 of {year}, GraphCast buffer

earth2studio variable → NetCDF mapping
---------------------------------------
Single-level (era5_single):
    u10m   → u10       v10m   → v10
    u100m  → u100      v100m  → v100
    t2m    → t2m       msl    → msl      tcwv → tcwv

Pressure-level (era5_pressure):
    u{p}   → u         v{p}   → v
    z{p}   → z         t{p}   → t        q{p} → q
    (where {p} is the pressure level in hPa, e.g. 500)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import _WORK

ERA5_DIR: Path = _WORK / "era5_6h"

# ── Variable mapping ──────────────────────────────────────────────────────────

_SINGLE_MAP: dict[str, str] = {
    "u10m":  "u10",
    "v10m":  "v10",
    "u100m": "u100",
    "v100m": "v100",
    "t2m":   "t2m",
    "msl":   "msl",
    "tcwv":  "tcwv",
}

_PRESSURE_PREFIX = {"u", "v", "z", "t", "q"}
_VALID_LEVELS    = {50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000}

_PRESSURE_RE = re.compile(r"^([uvztq])(\d+)$")


def _parse_variable(name: str) -> tuple[str, str | None, int | None]:
    """Return (file_type, nc_var, level) for an earth2studio variable name.

    file_type : "single" or "pressure"
    nc_var    : name in the NetCDF file
    level     : pressure level in hPa (None for single-level vars)

    Raises ValueError for unknown variable names.
    """
    if name in _SINGLE_MAP:
        return "single", _SINGLE_MAP[name], None

    m = _PRESSURE_RE.match(name)
    if m:
        prefix, level_str = m.group(1), m.group(2)
        if prefix in _PRESSURE_PREFIX:
            level = int(level_str)
            if level not in _VALID_LEVELS:
                raise ValueError(
                    f"Pressure level {level} hPa not available in local ERA5. "
                    f"Valid: {sorted(_VALID_LEVELS)}"
                )
            return "pressure", prefix, level

    raise ValueError(
        f"Variable '{name}' not recognised by LocalERA5. "
        f"Single-level vars: {list(_SINGLE_MAP)}. "
        f"Pressure-level vars: [u|v|z|t|q]{{level}}, e.g. u500, z250."
    )


# ── File resolution ───────────────────────────────────────────────────────────

def _single_path(t: datetime) -> list[Path]:
    """Candidate file paths for single-level data at time t."""
    paths = [ERA5_DIR / f"era5_single_{t.year}.nc"]
    if t.month == 12 and t.day == 31:
        paths.append(ERA5_DIR / f"era5_single_{t.year}_dec31.nc")
    # Also check dec31 file of previous year (t is Dec 31 of prev year)
    prev = t.year - 1
    paths.append(ERA5_DIR / f"era5_single_{prev}_dec31.nc")
    return paths


def _pressure_path(t: datetime) -> list[Path]:
    """Candidate file paths for pressure-level data at time t."""
    paths = [ERA5_DIR / f"era5_pressure_{t.year}_{t.month:02d}.nc"]
    if t.month == 12 and t.day == 31:
        paths.append(ERA5_DIR / f"era5_pressure_{t.year}_dec31.nc")
    prev = t.year - 1
    paths.append(ERA5_DIR / f"era5_pressure_{prev}_dec31.nc")
    return paths


def _find_file(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find local ERA5 file. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


# ── Dataset cache (one open handle per file) ──────────────────────────────────

_ds_cache: dict[Path, xr.Dataset] = {}


def _open(path: Path) -> xr.Dataset:
    if path not in _ds_cache:
        _ds_cache[path] = xr.open_dataset(str(path), engine="netcdf4")
    return _ds_cache[path]


# ── Time selection helpers ────────────────────────────────────────────────────

def _select_time(ds: xr.Dataset, t: datetime) -> xr.Dataset:
    """Select a single timestamp from an open dataset, flexible on coord name."""
    ts = pd.Timestamp(t)
    for dim in ("time", "valid_time"):
        if dim in ds.coords:
            return ds.sel({dim: ts}, method="nearest", tolerance=pd.Timedelta("1h"))
    raise KeyError(f"No time coordinate found in {ds}")


def _lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return the latitude and longitude dimension names used in ds."""
    for lat in ("latitude", "lat"):
        for lon in ("longitude", "lon"):
            if lat in ds.coords and lon in ds.coords:
                return lat, lon
    raise KeyError("Cannot find latitude/longitude coordinates in dataset.")


# ── Main class ────────────────────────────────────────────────────────────────

class LocalERA5:
    """earth2studio-compatible data source reading local ERA5 NetCDF files.

    Parameters
    ----------
    era5_dir : Path, optional
        Root directory containing the ERA5 NetCDF files.
        Defaults to config.ERA5_DIR.

    Usage
    -----
        data = LocalERA5()
        da = data(["2016-01-01T00:00:00"], ["u10m", "v10m", "u500", "z500"])
    """

    def __init__(self, era5_dir: Path | None = None) -> None:
        self._dir = Path(era5_dir) if era5_dir else ERA5_DIR
        if not self._dir.exists():
            raise FileNotFoundError(f"ERA5 directory not found: {self._dir}")

    def __call__(
        self,
        time: datetime | list[datetime] | np.ndarray,
        variable: str | list[str] | np.ndarray,
    ) -> xr.DataArray:
        """Fetch ERA5 data from local files.

        Parameters
        ----------
        time : datetime | list[datetime] | np.ndarray of datetime64
            UTC timestamps to fetch.
        variable : str | list[str] | np.ndarray
            earth2studio variable names (e.g. "u10m", "v10m", "u500").

        Returns
        -------
        xr.DataArray
            Shape (time, variable, lat, lon), consistent with the earth2studio
            DataSource protocol.
        """
        times, variables = _prep_inputs(time, variable)
        slices = [self._fetch_one(t, variables) for t in times]
        return xr.concat(slices, dim="time")

    def _fetch_one(self, t: datetime, variables: list[str]) -> xr.DataArray:
        """Return DataArray (variable, lat, lon) for a single timestamp."""
        arrays: list[xr.DataArray] = []

        for var in variables:
            file_type, nc_var, level = _parse_variable(var)

            if file_type == "single":
                path = _find_file(_single_path(t))
                ds   = _open(path)
                snap = _select_time(ds, t)
                lat_dim, lon_dim = _lat_lon_names(snap)
                da = snap[nc_var].rename({lat_dim: "lat", lon_dim: "lon"})

            else:  # pressure level
                path = _find_file(_pressure_path(t))
                ds   = _open(path)
                snap = _select_time(ds, t)
                lat_dim, lon_dim = _lat_lon_names(snap)

                # Select pressure level — coordinate may be named "level" or "pressure_level"
                for plev_dim in ("pressure_level", "level"):
                    if plev_dim in snap.coords:
                        da = snap[nc_var].sel({plev_dim: level}, method="nearest")
                        da = da.rename({lat_dim: "lat", lon_dim: "lon"})
                        break
                else:
                    raise KeyError(
                        f"Cannot find pressure-level coordinate in {path}. "
                        "Expected 'pressure_level' or 'level'."
                    )

            # Normalise to float32 and drop any leftover scalar coords
            da = da.astype(np.float32).squeeze(drop=True)
            da = da.expand_dims({"variable": [var]})
            arrays.append(da)

        out = xr.concat(arrays, dim="variable")

        # Ensure lat is ascending (some ERA5 files have lat descending)
        if float(out["lat"][0]) > float(out["lat"][-1]):
            out = out.isel(lat=slice(None, None, -1))

        out = out.expand_dims({"time": [np.datetime64(t, "ns")]})
        return out


# ── Input normalisation (mirrors earth2studio.data.utils.prep_data_inputs) ───

def _prep_inputs(
    time: datetime | list[datetime] | np.ndarray,
    variable: str | list[str] | np.ndarray,
) -> tuple[list[datetime], list[str]]:
    if isinstance(variable, str):
        variable = [variable]
    if isinstance(variable, np.ndarray):
        variable = list(variable.astype(str))

    if isinstance(time, datetime):
        time = [time]
    elif isinstance(time, np.ndarray):
        time = [pd.Timestamp(t).to_pydatetime() for t in time]
    elif isinstance(time, list):
        time = [
            pd.Timestamp(t).to_pydatetime() if not isinstance(t, datetime) else t
            for t in time
        ]
    return time, list(variable)
