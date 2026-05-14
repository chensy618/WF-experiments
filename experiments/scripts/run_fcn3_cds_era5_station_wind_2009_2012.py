from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import shutil

import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.models.px import FCN3
from earth2studio.data import CDS
from earth2studio.io import ZarrBackend
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
        "operating_period": "21.01.2003 - now",
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
        "operating_period": "01.11.1979 - now",
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
        "operating_period": "30.09.1964 - now",
        "wmo_number": "1025",
        "wigos_number": "0-20000-0-01025",
        "station_holder": "Avinor",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FCN3 with CDS ERA5 input and extract station-level 10 m wind speed."
        )
    )

    parser.add_argument(
        "--start",
        type=str,
        default="2009-01-01 00:00",
        help="Start datetime. Use 6-hourly timestamps for FCN3 experiments.",
    )

    parser.add_argument(
        "--end",
        type=str,
        default="2012-12-31 18:00",
        help="End datetime, inclusive. Use 6-hourly timestamps for FCN3 experiments.",
    )

    parser.add_argument(
        "--nsteps",
        type=int,
        default=1,
        help="Number of forecast steps.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/outputs",
        help="Output directory for final Zarr file.",
    )

    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep temporary monthly raw global Zarr files.",
    )

    return parser.parse_args()


def make_safe_output_path(output_dir: Path, start: str, end: str, nsteps: int) -> Path:
    """
    Create a unique output path so old Zarr files are never overwritten.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_tag = pd.Timestamp(start).strftime("%Y%m%d%H")
    end_tag = pd.Timestamp(end).strftime("%Y%m%d%H")

    output_path = output_dir / (
        f"fcn3_cds_era5_station_wind_"
        f"{start_tag}_{end_tag}_nsteps{nsteps}_{timestamp}.zarr"
    )

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    return output_path


def get_monthly_time_chunks(start: str, end: str) -> list[list[str]]:
    """
    Split 6-hourly timestamps into monthly chunks.

    CDS can be slow, so monthly chunks are safer than one huge request.
    """
    all_times = pd.date_range(start=start, end=end, freq="6h")

    chunks = []
    for _, month_times in all_times.to_series().groupby(all_times.to_period("M")):
        chunks.append([t.strftime("%Y-%m-%d %H:%M") for t in month_times])

    return chunks


def extract_wind_components(ds: xr.Dataset):
    """
    Extract 10 m wind components from Earth2Studio output.

    Supports two possible layouts:
    1. u/v are separate data variables, e.g. ds["u10"], ds["v10"].
    2. u/v are stored inside one large array with a variable/channel dimension.
    """

    if "u10" in ds.data_vars and "v10" in ds.data_vars:
        return ds["u10"], ds["v10"], "u10", "v10"

    if "u10m" in ds.data_vars and "v10m" in ds.data_vars:
        return ds["u10m"], ds["v10m"], "u10m", "v10m"

    main_var = max(ds.data_vars, key=lambda name: ds[name].size)
    arr = ds[main_var]

    possible_var_dims = ["variable", "variables", "channel", "var"]

    for dim in possible_var_dims:
        if dim in arr.dims and dim in ds.coords:
            names = [str(v) for v in ds[dim].values]

            if "u10" in names and "v10" in names:
                return arr.sel({dim: "u10"}), arr.sel({dim: "v10"}), "u10", "v10"

            if "u10m" in names and "v10m" in names:
                return arr.sel({dim: "u10m"}), arr.sel({dim: "v10m"}), "u10m", "v10m"

    raise ValueError(
        "Could not find 10 m wind variables.\n"
        f"Data variables: {list(ds.data_vars)}\n"
        f"Coordinates: {list(ds.coords)}\n"
        f"Main array dims: {arr.dims}"
    )


def extract_station_wind(ds: xr.Dataset) -> xr.Dataset:
    """
    Extract station-nearest 10 m wind speed from one monthly FCN3 output.
    """
    u_field, v_field, u_name, v_name = extract_wind_components(ds)

    station_outputs = []

    for station in STATIONS:
        u_station = u_field.sel(
            lat=station["lat"],
            lon=station["lon"],
            method="nearest",
        )

        v_station = v_field.sel(
            lat=station["lat"],
            lon=station["lon"],
            method="nearest",
        )

        wind_speed = np.sqrt(u_station**2 + v_station**2)
        wind_speed.name = "wind_speed_10m"

        wind_speed = wind_speed.expand_dims(station=[station["station_id"]])

        wind_speed = wind_speed.assign_coords(
            station_name=("station", [station["station_name"]]),
            municipality=("station", [station["municipality"]]),
            county=("station", [station["county"]]),
            station_height_m=("station", [station["height_m"]]),
            station_lat=("station", [station["lat"]]),
            station_lon=("station", [station["lon"]]),
            operating_period=("station", [station["operating_period"]]),
            wmo_number=("station", [station["wmo_number"]]),
            wigos_number=("station", [station["wigos_number"]]),
            station_holder=("station", [station["station_holder"]]),
            nearest_grid_lat=("station", [float(u_station["lat"].values)]),
            nearest_grid_lon=("station", [float(u_station["lon"].values)]),
        )

        station_outputs.append(wind_speed)

    wind_all = xr.concat(station_outputs, dim="station")

    out = wind_all.to_dataset(name="wind_speed_10m")

    out["wind_speed_10m"].attrs["long_name"] = "10 m wind speed"
    out["wind_speed_10m"].attrs["units"] = "m/s"
    out["wind_speed_10m"].attrs["computed_from"] = f"sqrt({u_name}^2 + {v_name}^2)"

    out.attrs["model"] = "FCN3"
    out.attrs["input_data"] = "CDS ERA5"
    out.attrs["description"] = (
        "Station-nearest 10 m wind speed prediction extracted from FCN3 forecasts "
        "initialized with ERA5 data from the CDS API."
    )

    return out


def print_station_summary() -> None:
    print("Stations:")
    for station in STATIONS:
        print(
            f"  {station['station_id']} | "
            f"{station['station_name']} | "
            f"lat={station['lat']} | "
            f"lon={station['lon']} | "
            f"height={station['height_m']} m"
        )


def main() -> None:
    args = parse_args()

    print("============================================================")
    print("FCN3 CDS ERA5 station wind prediction")
    print("============================================================")

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / "tmp_fcn3_cds_station_wind_raw"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    final_zarr_path = make_safe_output_path(
        output_dir=output_dir,
        start=args.start,
        end=args.end,
        nsteps=args.nsteps,
    )

    print("Start:", args.start)
    print("End:", args.end)
    print("Forecast steps:", args.nsteps)
    print("Final output:", final_zarr_path)

    print_station_summary()

    print("Loading FCN3 package...")
    package = FCN3.load_default_package()

    print("Loading FCN3 model...")
    model = FCN3.load_model(package)

    print("Loading CDS ERA5 data source...")
    print("This requires a valid ~/.cdsapirc or CDS API configuration.")
    data = CDS(cache=True, verbose=True)

    monthly_chunks = get_monthly_time_chunks(args.start, args.end)
    print("Number of monthly chunks:", len(monthly_chunks))

    monthly_station_datasets = []

    for chunk_idx, times in enumerate(monthly_chunks, start=1):
        first_time = pd.Timestamp(times[0])
        month_tag = first_time.strftime("%Y%m")

        raw_zarr_path = tmp_dir / f"raw_fcn3_cds_era5_{month_tag}.zarr"

        if raw_zarr_path.exists():
            print(f"Removing old temporary raw output: {raw_zarr_path}")
            shutil.rmtree(raw_zarr_path)

        print("------------------------------------------------------------")
        print(f"Chunk {chunk_idx}/{len(monthly_chunks)}")
        print("Month:", month_tag)
        print("Number of initial times:", len(times))
        print("First time:", times[0])
        print("Last time:", times[-1])
        print("Temporary raw Zarr:", raw_zarr_path)

        io = ZarrBackend(file_name=str(raw_zarr_path))

        deterministic(
            times,
            args.nsteps,
            model,
            data,
            io,
        )

        print("Opening raw monthly output...")
        ds = xr.open_zarr(raw_zarr_path)

        print("Extracting station wind speed...")
        station_ds = extract_station_wind(ds)

        monthly_station_datasets.append(station_ds.load())

        ds.close()

        if not args.keep_raw:
            print("Deleting temporary raw Zarr:", raw_zarr_path)
            shutil.rmtree(raw_zarr_path)

    print("============================================================")
    print("Combining monthly station outputs")
    print("============================================================")

    combined = xr.concat(monthly_station_datasets, dim="time")

    if "time" in combined.coords:
        combined = combined.sortby("time")

    combined.attrs["start_time"] = args.start
    combined.attrs["end_time"] = args.end
    combined.attrs["nsteps"] = args.nsteps
    combined.attrs["created_at"] = datetime.now().isoformat()
    combined.attrs["note"] = (
        "This file contains only station-nearest extracted wind speed values, "
        "not full global forecast fields. The forecast was initialized with CDS ERA5 data."
    )

    print("Final dataset:")
    print(combined)

    print("Saving final station wind Zarr...")
    combined.to_zarr(final_zarr_path, mode="w")

    print("============================================================")
    print("Finished successfully")
    print("Saved final output to:")
    print(final_zarr_path)
    print("============================================================")


if __name__ == "__main__":
    main()