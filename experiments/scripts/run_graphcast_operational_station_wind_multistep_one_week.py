"""
Run GraphCast Operational with 6-hourly cycling initialization and extract
station-level 10 m wind speed.

Multi-step version:
  - 6-hourly inits: 00:00, 06:00, 12:00, 18:00 UTC each day
  - nsteps = 4  -> +6h, +12h, +18h, +24h
  - Output: compact zarr (station x time x lead_time)
  - Raw global zarr files are temporary and deleted by default
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from earth2studio.data import CDS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import GraphCastOperational
from earth2studio.run import deterministic


STATIONS = [
    {
        "station_id": "SN90760",
        "station_name": "Fakken",
        "municipality": "Karlsøy",
        "county": "Troms",
        "height_m": 57.0,
        "lat": 70.10426,
        "lon": 20.11451,
        "wmo_number": "",
        "wigos_number": "0-578-0-90760",
        "station_holder": "Met.no",
    },
    {
        "station_id": "SN88690",
        "station_name": "Hekkingen Fyr",
        "municipality": "Senja",
        "county": "Troms",
        "height_m": 33.0,
        "lat": 69.6005,
        "lon": 17.8317,
        "wmo_number": "1015",
        "wigos_number": "0-20000-0-01015",
        "station_holder": "Met.no",
    },
    {
        "station_id": "SN90490",
        "station_name": "Tromsø-Langnes",
        "municipality": "Tromsø",
        "county": "Troms",
        "height_m": 8.0,
        "lat": 69.6767,
        "lon": 18.9133,
        "wmo_number": "1025",
        "wigos_number": "0-20000-0-01025",
        "station_holder": "Avinor",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GraphCast Operational with CDS ERA5 input and extract "
            "station-level 10 m wind speed for multi-step forecasts."
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
        help="Number of days in this weekly chunk.",
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
        help="Directory where weekly station zarr will be saved.",
    )

    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep temporary raw global zarr files for debugging.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output if present.",
    )

    return parser.parse_args()


def _nearest_idx(
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[int, int]:
    lat_i = int(np.argmin(np.abs(lats - target_lat)))
    lon_i = int(np.argmin(np.abs(lons - (target_lon % 360))))
    return lat_i, lon_i


def _build_stn_idx(lats: np.ndarray, lons: np.ndarray) -> dict[str, tuple[int, int]]:
    stn_idx: dict[str, tuple[int, int]] = {}

    for st in STATIONS:
        lat_i, lon_i = _nearest_idx(lats, lons, st["lat"], st["lon"])
        stn_idx[st["station_id"]] = (lat_i, lon_i)

        print(
            f"  {st['station_id']} ({st['station_name']}): "
            f"target=({st['lat']:.4f}N, {st['lon']:.4f}E)  "
            f"grid=({lats[lat_i]:.2f}N, {lons[lon_i]:.2f}E)"
        )

    return stn_idx


def _decode_times(g: zarr.Group) -> pd.DatetimeIndex:
    time_raw = g["time"][:]

    if np.issubdtype(time_raw.dtype, np.datetime64):
        return pd.DatetimeIndex(time_raw)

    return pd.DatetimeIndex([pd.Timestamp(int(t), unit="ns") for t in time_raw])


def _decode_lead_hours(g: zarr.Group) -> np.ndarray:
    lt_raw = g["lead_time"][:]

    if np.issubdtype(lt_raw.dtype, np.timedelta64):
        return (lt_raw / np.timedelta64(1, "h")).astype(float)

    return lt_raw.astype(float)


def extract_station_wind(
    raw_zarr_path: Path,
    stn_idx: dict[str, tuple[int, int]] | None,
) -> tuple[xr.Dataset, dict[str, tuple[int, int]]]:
    """
    Extract station-nearest 10 m wind speed from one raw GraphCast global zarr.

    Returns:
      - station-level xarray Dataset
      - station-to-grid-index map
    """
    g = zarr.open_group(str(raw_zarr_path), mode="r")

    lats = g["lat"][:]
    lons = g["lon"][:]

    if stn_idx is None:
        print("  Building station-to-grid-index map:")
        stn_idx = _build_stn_idx(lats, lons)

    init_times = _decode_times(g)
    lead_hours = _decode_lead_hours(g)

    u10m = g["u10m"][:]
    v10m = g["v10m"][:]

    wind_speed = np.sqrt(u10m**2 + v10m**2).astype(np.float32)

    station_das = []

    for st in STATIONS:
        sid = st["station_id"]
        lat_i, lon_i = stn_idx[sid]

        ws_stn = wind_speed[:, :, lat_i, lon_i]

        da = xr.DataArray(
            ws_stn,
            dims=["time", "lead_time"],
            coords={
                "time": init_times,
                "lead_time": lead_hours.astype("timedelta64[h]"),
            },
        )

        da = da.expand_dims(station=[sid])

        da = da.assign_coords(
            station_name=("station", [st["station_name"]]),
            municipality=("station", [st["municipality"]]),
            county=("station", [st["county"]]),
            station_height_m=("station", [st["height_m"]]),
            station_lat=("station", [st["lat"]]),
            station_lon=("station", [st["lon"]]),
            wmo_number=("station", [st["wmo_number"]]),
            wigos_number=("station", [st["wigos_number"]]),
            station_holder=("station", [st["station_holder"]]),
            nearest_grid_lat=("station", [float(lats[lat_i])]),
            nearest_grid_lon=("station", [float(lons[lon_i])]),
        )

        station_das.append(da)

    wind_all = xr.concat(station_das, dim="station")

    out = wind_all.to_dataset(name="wind_speed_10m")

    out["wind_speed_10m"].attrs.update(
        long_name="10 m wind speed",
        units="m/s",
        computed_from="sqrt(u10m^2 + v10m^2)",
    )

    out.attrs.update(
        model="GraphCastOperational",
        input_data="CDS ERA5",
        init_freq="6h",
        output_type="station_nearest_gridpoint",
    )

    return out, stn_idx


def main() -> None:
    args = parse_args()

    week_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    week_end = week_start + timedelta(days=args.days - 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / "tmp_graphcast_raw_multistep"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    start_tag = week_start.strftime("%Y%m%d")
    end_tag = week_end.strftime("%Y%m%d")

    out_zarr = output_dir / f"graphcast_station_wind_multistep_{start_tag}_{end_tag}.zarr"

    lead_hours = [6 * i for i in range(1, args.nsteps + 1)]

    print("=" * 80)
    print("GraphCast Operational station wind multi-step forecast")
    print(f"Week start:  {week_start.strftime('%Y-%m-%d')}")
    print(f"Week end:    {week_end.strftime('%Y-%m-%d')}")
    print(f"Days:        {args.days}")
    print(f"N steps:     {args.nsteps}")
    print(f"Lead hours:  {lead_hours}")
    print(f"Output:      {out_zarr}")
    print("=" * 80)

    if out_zarr.exists():
        if args.overwrite:
            print(f"Removing existing output: {out_zarr}")
            shutil.rmtree(out_zarr)
        else:
            print("Output already exists. Skipping.")
            print(f"Existing output: {out_zarr}")
            return

    print("Loading GraphCastOperational package...")
    package = GraphCastOperational.load_default_package()

    print("Loading GraphCastOperational model...")
    model = GraphCastOperational.load_model(package)

    print("Loading CDS ERA5 data source...")
    data = CDS(cache=True, verbose=True)

    stn_idx = None
    day_parts = []

    for day_offset in range(args.days):
        day = week_start + timedelta(days=day_offset)
        day_tag = day.strftime("%Y%m%d")

        day_times = [
            t.strftime("%Y-%m-%dT%H:%M:%S")
            for t in pd.date_range(
                start=day.strftime("%Y-%m-%d 00:00"),
                end=day.strftime("%Y-%m-%d 18:00"),
                freq="6h",
            )
        ]

        raw_zarr = tmp_dir / f"graphcast_raw_multistep_{day_tag}.zarr"

        if raw_zarr.exists():
            shutil.rmtree(raw_zarr)

        print("-" * 80)
        print(f"[Day {day_offset + 1}/{args.days}] {day.strftime('%Y-%m-%d')}")
        print(f"Init times: {day_times}")
        print(f"Temporary raw zarr: {raw_zarr}")
        print("-" * 80)

        io = ZarrBackend(str(raw_zarr))

        deterministic(
            day_times,
            args.nsteps,
            model,
            data,
            io,
        )

        print(f"Extracting station wind from {raw_zarr.name} ...")
        stn_ds, stn_idx = extract_station_wind(raw_zarr, stn_idx)
        day_parts.append(stn_ds.load())

        if not args.keep_raw:
            shutil.rmtree(raw_zarr)
            print(f"Deleted temporary raw zarr: {raw_zarr.name}")

    print("=" * 80)
    print("Combining daily station outputs...")
    combined = xr.concat(day_parts, dim="time").sortby("time")

    combined.attrs.update(
        model="GraphCastOperational",
        input_data="CDS ERA5",
        week_start=args.start_date,
        week_end=week_end.strftime("%Y-%m-%d"),
        nsteps=args.nsteps,
        lead_hours=",".join(str(h) for h in lead_hours),
        init_freq="6h",
        created_at=datetime.now().isoformat(),
        description=(
            "Station-nearest 10 m wind speed from GraphCast Operational. "
            "The model is initialized every 6 hours and rolled out for "
            "+6h,+12h,+18h,+24h."
        ),
    )

    print(f"Saving compact station zarr to {out_zarr} ...")
    combined.to_zarr(out_zarr, mode="w")

    success_file = out_zarr / "_SUCCESS"
    success_file.write_text(
        f"model=GraphCastOperational\n"
        f"week_start={args.start_date}\n"
        f"week_end={week_end.strftime('%Y-%m-%d')}\n"
        f"nsteps={args.nsteps}\n"
        f"lead_hours={','.join(str(h) for h in lead_hours)}\n"
        f"init_freq=6h\n"
    )

    print("=" * 80)
    print("Finished GraphCast Operational station wind multi-step forecast.")
    print(f"Saved output: {out_zarr}")
    print("=" * 80)


if __name__ == "__main__":
    main()