"""
Data loaders: station observations, HRES (WeatherBench2), forecast zarrs.
All functions return tidy DataFrames with columns (station, valid_time, *_wind).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from config import HRES_PATH, LEAD_H, OBS_DIR, STATIONS, STATIONS_BY_ID


# =============================================================================
# Station observations
# =============================================================================

def load_observations(year: int) -> pd.DataFrame:
    """Load hourly Frost CSV obs for one year and compute 12-h trailing mean.

    Returns columns: station, valid_time, obs_wind
    """
    parts = []
    for st in STATIONS:
        f = OBS_DIR / f"{st['obs_base']}_{year}.csv"
        if not f.exists():
            print(f"  [obs] Missing: {f.name}")
            continue
        df = pd.read_csv(f)
        df["valid_time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        df["station"] = st["id"]
        df = df.rename(columns={"wind_speed": "obs_wind"})
        df = df[["station", "valid_time", "obs_wind"]].dropna(subset=["obs_wind"])
        df["obs_wind"] = (
            df.set_index("valid_time")["obs_wind"]
            .rolling("12h", min_periods=6)
            .mean()
            .values
        )
        parts.append(df.dropna(subset=["obs_wind"]))

    if not parts:
        return pd.DataFrame(columns=["station", "valid_time", "obs_wind"])
    return pd.concat(parts, ignore_index=True).sort_values("valid_time").reset_index(drop=True)


# =============================================================================
# HRES (WeatherBench2 zarr)
# =============================================================================

def load_hres(year: int) -> pd.DataFrame:
    """Extract HRES nearest-gridpoint 10 m wind at LEAD_H for all stations.

    Returns columns: station, valid_time, hres_wind
    """
    if not HRES_PATH.exists():
        print(f"  [HRES] Not found: {HRES_PATH}")
        return pd.DataFrame()

    ds = xr.open_zarr(str(HRES_PATH), consolidated=False)
    da = ds["10m_wind_speed"].sel(prediction_timedelta=LEAD_H)
    da = da.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    init_times = pd.DatetimeIndex(da["time"].values)

    rows = []
    for st in STATIONS:
        ws_vals = da.sel(
            latitude=st["lat"], longitude=st["lon"] % 360, method="nearest"
        ).values
        for t_i, init_t in enumerate(init_times):
            val = float(ws_vals[t_i])
            if not np.isnan(val):
                rows.append({
                    "station":    st["id"],
                    "valid_time": init_t + pd.Timedelta(hours=LEAD_H),
                    "hres_wind":  val,
                })

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["station", "valid_time"])
        .sort_values(["station", "valid_time"])
        .reset_index(drop=True)
    )


# =============================================================================
# Model forecast zarrs (FCN3 / GraphCast — compact station zarrs)
# =============================================================================

def load_forecast(fc_dir: Path, model_label: str) -> pd.DataFrame:
    """Concatenate all station zarrs in fc_dir into a tidy DataFrame.

    Returns columns: station, valid_time, {model_label.lower()}_wind
    """
    col   = f"{model_label.lower()}_wind"
    files = sorted(fc_dir.glob(f"{model_label.lower()}_*.zarr"))
    if not files:
        print(f"  [{model_label}] No forecast files in {fc_dir}")
        return pd.DataFrame()

    parts = []
    for f in files:
        ds        = xr.open_zarr(str(f))
        sids      = [str(s) for s in ds["station"].values]
        init_times = pd.DatetimeIndex(ds["time"].values)
        lt_raw    = ds["lead_time"].values
        lt_hours  = (lt_raw / np.timedelta64(1, "h")).astype(float)
        lt_idx    = int(np.argmin(np.abs(lt_hours - LEAD_H)))
        lt_used   = float(lt_hours[lt_idx])
        ws        = ds["wind_speed_10m"].values  # (station, time, lead)

        for s_i, sid in enumerate(sids):
            if sid not in STATIONS_BY_ID:
                continue
            for t_i, init_t in enumerate(init_times):
                wind = float(ws[s_i, t_i, lt_idx])
                if not np.isnan(wind):
                    parts.append({
                        "station":    sid,
                        "valid_time": init_t + pd.Timedelta(hours=lt_used),
                        col:          wind,
                    })

    if not parts:
        return pd.DataFrame()
    return (
        pd.DataFrame(parts)
        .drop_duplicates(subset=["station", "valid_time"])
        .sort_values(["station", "valid_time"])
        .reset_index(drop=True)
    )


# =============================================================================
# Merge helper
# =============================================================================

def build_merged(year: int, fc_dir: Path, model_label: str) -> pd.DataFrame:
    """Inner-join obs, HRES, and model forecast for one year.

    Returns a DataFrame with columns:
        station, valid_time, obs_wind, hres_wind, {model}_wind, year, period
    """
    from config import IN_TRAINING_YEARS

    obs_df  = load_observations(year)
    hres_df = load_hres(year)
    fc_df   = load_forecast(fc_dir, model_label)

    for df in [obs_df, hres_df, fc_df]:
        if not df.empty:
            df["valid_time"] = df["valid_time"].astype("datetime64[us]")

    merged = obs_df.copy() if not obs_df.empty else pd.DataFrame()
    for df, col in [(hres_df, "hres_wind"), (fc_df, f"{model_label.lower()}_wind")]:
        if df.empty:
            continue
        merged = (
            pd.merge(merged, df[["station", "valid_time", col]],
                     on=["station", "valid_time"], how="inner")
            if not merged.empty else df
        )

    if merged.empty:
        return pd.DataFrame()

    merged["year"]   = year
    merged["period"] = "in-training" if year in IN_TRAINING_YEARS else "out-of-training"
    return merged.reset_index(drop=True)
