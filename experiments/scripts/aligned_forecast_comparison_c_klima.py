# # Aligned comparison: FCN3, GraphCast, Pangu, HRES, and C-klima observations
# 
# This notebook compares the currently available aligned forecast periods:
# 
# | Experiment | Models / data | Lead time | Period |
# |---|---|---:|---|
# | 2016 multistep | FCN3 vs HRES vs OBS | +12h | 2016-01-01 to 2016-05-12 |
# | 2018 multistep | FCN3 vs HRES vs OBS | +12h | 2018-01-01 to 2018-05-20 |
# | 2022 aligned | FCN3 vs GraphCast vs Pangu vs HRES vs OBS | +12h | 2022-01-01 to 2022-01-28 |
# 
# OBS is downloaded from C-klima / MET observation data through the Frost API.

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import requests
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

# ## 1. Configuration
# 
# Update only the paths marked `TODO` if your FCN3 / GraphCast output Zarr paths are different.
# 
# The HRES and Pangu paths are filled using the folders you showed earlier.

PROJECT_ROOT = Path("/cluster/work/projects/nn8106k/siyan")
FORECAST_ROOT = PROJECT_ROOT / "weatherbench2_forecasts"

# Reference / downloaded forecast datasets
HRES_0P25 = FORECAST_ROOT / "hres/0p25/2016-2022-0012-1440x721.zarr"
PANGU_0P25 = FORECAST_ROOT / "pangu/0p25/2018-2022_0012_0p25.zarr"

# TODO: set these to your actual output zarr stores.
# You can use the helper cell below to search candidate .zarr directories.
FCN3_2016_ZARR = Path("/cluster/work/projects/nn8106k/siyan/CHANGE_ME/fcn3_weekly_multistep_2016.zarr")
FCN3_2018_ZARR = Path("/cluster/work/projects/nn8106k/siyan/CHANGE_ME/fcn3_weekly_multistep_2018.zarr")
FCN3_2022_ZARR = Path("/cluster/work/projects/nn8106k/siyan/CHANGE_ME/fcn3_weekly_2022_2025_single_step_6h.zarr")

GRAPHCAST_2022_ZARR = Path("/cluster/work/projects/nn8106k/siyan/CHANGE_ME/graphcast_weekly_2022_2025_single_step_6h.zarr")

# Stations to compare. Replace / extend this list with the stations used in your previous notebook.
STATION_IDS = [
    "SN88690",
    "SN90490",
    "SN90760",
]

LEAD_HOURS = 12
OBS_ELEMENTS = [
    "wind_speed",
    "max(wind_speed PT1H)",
    "wind_from_direction",
]

OUTPUT_DIR = Path("./aligned_comparison_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("Output dir:", OUTPUT_DIR.resolve())

def find_zarrs(root: Path, keyword: str, max_results: int = 30):
    root = Path(root)
    out = []
    for p in root.rglob("*.zarr"):
        if keyword.lower() in str(p).lower():
            out.append(p)
            if len(out) >= max_results:
                break
    return out

# Uncomment this cell if you need to locate your generated FCN3 / GraphCast zarr stores.
# for kw in [
#     "fcn3_weekly_multistep_2016",
#     "fcn3_weekly_multistep_2018",
#     "fcn3_weekly_2022",
#     "graphcast_weekly_2022",
# ]:
#     print("\\n==", kw, "==")
#     for p in find_zarrs(PROJECT_ROOT, kw):
#         print(p)

# ## 2. Download C-klima / Frost observations
# 
# Set your Frost client ID first:
# 
# ```bash
# export FROST_CLIENT_ID="your_client_id"
# ```
# 
# The code saves observations locally so they are not downloaded repeatedly.

FROST_BASE = "https://frost.met.no"

def _get_frost_client_id():
    client_id = os.environ.get("FROST_CLIENT_ID")
    if not client_id:
        raise RuntimeError(
            "Missing FROST_CLIENT_ID. Run this before opening Jupyter:\n"
            'export FROST_CLIENT_ID="your_client_id"'
        )
    return client_id

def frost_get(endpoint: str, params: dict):
    client_id = _get_frost_client_id()
    url = f"{FROST_BASE}{endpoint}"
    r = requests.get(url, params=params, auth=(client_id, ""))
    if not r.ok:
        print("URL:", r.url)
        print("Response:", r.text[:1000])
        r.raise_for_status()
    return r.json()

def download_station_metadata(station_ids):
    params = {
        "sources": ",".join(station_ids),
        "fields": "id,name,geometry,masl",
    }
    data = frost_get("/sources/v0.jsonld", params)
    rows = []
    for item in data.get("data", []):
        lon, lat = item["geometry"]["coordinates"][:2]
        rows.append({
            "station_id": item["id"],
            "name": item.get("name"),
            "lat": lat,
            "lon": lon,
            "masl": item.get("masl"),
        })
    return pd.DataFrame(rows)

def download_observations(station_ids, start_date, end_date, elements=OBS_ELEMENTS, cache_dir=OUTPUT_DIR / "obs_cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    safe_sources = "-".join(station_ids)
    safe_elements = "_".join([e.replace("(", "").replace(")", "").replace(" ", "_").replace("/", "_") for e in elements])
    cache_file = cache_dir / f"obs_{safe_sources}_{start_date}_{end_date}_{safe_elements}.csv"

    if cache_file.exists():
        print("Loading cached observations:", cache_file)
        return pd.read_csv(cache_file, parse_dates=["time"])

    params = {
        "sources": ",".join(station_ids),
        "referencetime": f"{start_date}/{end_date}",
        "elements": ",".join(elements),
    }

    data = frost_get("/observations/v0.jsonld", params)

    rows = []
    for rec in data.get("data", []):
        source_id = rec.get("sourceId", "").split(":")[0]
        ref_time = pd.to_datetime(rec["referenceTime"]).tz_localize(None)

        for obs in rec.get("observations", []):
            rows.append({
                "station_id": source_id,
                "time": ref_time,
                "element": obs.get("elementId"),
                "value": obs.get("value"),
                "unit": obs.get("unit"),
                "quality_code": obs.get("qualityCode"),
            })

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError(f"No observations returned for {station_ids}, {start_date} to {end_date}")

    wide = (
        long_df
        .pivot_table(index=["station_id", "time"], columns="element", values="value", aggfunc="first")
        .reset_index()
    )
    wide.columns.name = None

    if "wind_speed" in wide.columns:
        wide["obs_wind_speed"] = wide["wind_speed"]
    elif "max(wind_speed PT1H)" in wide.columns:
        wide["obs_wind_speed"] = wide["max(wind_speed PT1H)"]
    else:
        raise ValueError("No usable wind speed element found. Expected wind_speed or max(wind_speed PT1H).")

    wide.to_csv(cache_file, index=False)
    print("Saved observations:", cache_file)
    return wide

# ## 3. Forecast loading and station extraction helpers
# 
# The helper functions below support common WeatherBench2 / Earth2Studio coordinate names:
# 
# - `time` or `forecast_reference_time`
# - `lead_time` or `prediction_timedelta`
# - `lat` / `latitude`
# - `lon` / `longitude`
# 
# For wind speed, the code first looks for a direct wind-speed variable. If not found, it computes:
# 
# $$	ext{wind speed} = \sqrt{u_{10}^2 + v_{10}^2}$$

def open_zarr(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {path}")
    return xr.open_zarr(path, consolidated=False)

def find_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims or c in ds.variables:
            return c
    raise KeyError(f"Could not find any of {candidates}. Available coords/dims: {list(ds.coords)} / {list(ds.dims)}")

def get_time_coord(ds):
    return find_coord_name(ds, ["time", "forecast_reference_time", "init_time"])

def get_lead_coord(ds):
    for c in ["lead_time", "prediction_timedelta", "step", "lead"]:
        if c in ds.coords or c in ds.dims:
            return c
    return None

def get_lat_lon_names(ds):
    lat_name = find_coord_name(ds, ["lat", "latitude"])
    lon_name = find_coord_name(ds, ["lon", "longitude"])
    return lat_name, lon_name

def select_lead(ds, lead_hours=12):
    lead_name = get_lead_coord(ds)
    if lead_name is None:
        print("No lead_time / prediction_timedelta coordinate found. Assuming file already stores one lead time.")
        return ds, None

    lead_values = ds[lead_name].values
    target = np.timedelta64(lead_hours, "h")

    if np.issubdtype(ds[lead_name].dtype, np.timedelta64):
        if target in lead_values:
            return ds.sel({lead_name: target}), target
        diffs = np.abs(lead_values - target)
        chosen = lead_values[int(np.argmin(diffs))]
        print(f"Exact {lead_hours}h lead not found. Using nearest lead:", chosen)
        return ds.sel({lead_name: chosen}), chosen

    vals = np.asarray(lead_values, dtype=float)
    idx = int(np.argmin(np.abs(vals - lead_hours)))
    chosen = lead_values[idx]
    if float(chosen) != float(lead_hours):
        print(f"Exact {lead_hours}h lead not found. Using nearest lead:", chosen)
    return ds.sel({lead_name: chosen}), chosen

def detect_wind_speed_da(ds):
    direct_candidates = [
        "wind_speed", "10m_wind_speed", "si10", "ws10", "wind_speed_10m", "10m_wind_speed_surface"
    ]
    for v in direct_candidates:
        if v in ds.data_vars:
            return ds[v]

    u_candidates = ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "u_component_of_wind"]
    v_candidates = ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "v_component_of_wind"]

    u_name = next((v for v in u_candidates if v in ds.data_vars), None)
    v_name = next((v for v in v_candidates if v in ds.data_vars), None)

    if u_name and v_name:
        return np.sqrt(ds[u_name] ** 2 + ds[v_name] ** 2).rename("wind_speed")

    raise KeyError("Could not detect wind-speed variable. Available data variables:\n" + "\n".join(list(ds.data_vars)))

def normalize_longitude_for_ds(lon, ds_lon_values):
    ds_lon_values = np.asarray(ds_lon_values)
    lon = float(lon)
    if np.nanmin(ds_lon_values) >= 0 and lon < 0:
        return lon % 360
    if np.nanmax(ds_lon_values) <= 180 and lon > 180:
        return ((lon + 180) % 360) - 180
    return lon

def extract_model_at_stations(zarr_path, model_name, stations_df, start_date, end_date, lead_hours=12):
    ds = open_zarr(zarr_path)
    ds_lead, chosen_lead = select_lead(ds, lead_hours=lead_hours)

    time_name = get_time_coord(ds_lead)
    lat_name, lon_name = get_lat_lon_names(ds_lead)
    da = detect_wind_speed_da(ds_lead)

    init_start = pd.to_datetime(start_date) - pd.Timedelta(hours=lead_hours)
    init_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    try:
        da = da.sel({time_name: slice(init_start, init_end)})
    except Exception:
        pass

    rows = []
    lon_values = ds_lead[lon_name].values

    for _, st in stations_df.iterrows():
        station_id = st["station_id"]
        lat = float(st["lat"])
        lon = normalize_longitude_for_ds(float(st["lon"]), lon_values)

        point = da.sel({lat_name: lat, lon_name: lon}, method="nearest")
        nearest_lat = float(point[lat_name].values)
        nearest_lon = float(point[lon_name].values)
        df = point.to_dataframe(name="forecast_wind_speed").reset_index()

        if chosen_lead is None:
            df["valid_time"] = pd.to_datetime(df[time_name])
        elif isinstance(chosen_lead, np.timedelta64):
            df["valid_time"] = pd.to_datetime(df[time_name]) + pd.to_timedelta(chosen_lead)
        else:
            df["valid_time"] = pd.to_datetime(df[time_name]) + pd.to_timedelta(float(chosen_lead), unit="h")

        df["station_id"] = station_id
        df["station_name"] = st.get("name", station_id)
        df["station_lat"] = lat
        df["station_lon"] = float(st["lon"])
        df["grid_lat"] = nearest_lat
        df["grid_lon"] = nearest_lon
        df["model"] = model_name
        df["lead_hours"] = lead_hours

        keep = [
            "station_id", "station_name", "valid_time", "model", "lead_hours",
            "forecast_wind_speed", "station_lat", "station_lon", "grid_lat", "grid_lon"
        ]
        rows.append(df[keep])

    out = pd.concat(rows, ignore_index=True)
    out["valid_time"] = pd.to_datetime(out["valid_time"]).dt.tz_localize(None)

    start = pd.to_datetime(start_date)
    end_exclusive = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    out = out[(out["valid_time"] >= start) & (out["valid_time"] < end_exclusive)].copy()
    return out

# ## 4. Metrics and plotting helpers

def align_with_obs(model_df, obs_df):
    obs = obs_df.copy()
    obs["time"] = pd.to_datetime(obs["time"]).dt.tz_localize(None)
    merged = model_df.merge(
        obs[["station_id", "time", "obs_wind_speed"]],
        left_on=["station_id", "valid_time"],
        right_on=["station_id", "time"],
        how="inner",
    )
    merged = merged.drop(columns=["time"])
    merged["error"] = merged["forecast_wind_speed"] - merged["obs_wind_speed"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2
    return merged

def compute_metrics(df):
    rows = []
    for (model, station_id), g in df.groupby(["model", "station_id"]):
        rows.append({
            "model": model,
            "station_id": station_id,
            "N": len(g),
            "RMSE": np.sqrt(np.mean(g["squared_error"])),
            "MAE": np.mean(g["abs_error"]),
            "Bias": np.mean(g["error"]),
            "Obs mean": np.mean(g["obs_wind_speed"]),
            "Forecast mean": np.mean(g["forecast_wind_speed"]),
        })
    by_station = pd.DataFrame(rows).sort_values(["station_id", "model"])

    rows = []
    for model, g in df.groupby("model"):
        rows.append({
            "model": model,
            "N": len(g),
            "RMSE": np.sqrt(np.mean(g["squared_error"])),
            "MAE": np.mean(g["abs_error"]),
            "Bias": np.mean(g["error"]),
            "Obs mean": np.mean(g["obs_wind_speed"]),
            "Forecast mean": np.mean(g["forecast_wind_speed"]),
        })
    overall = pd.DataFrame(rows).sort_values("model")
    return by_station, overall

def plot_timeseries(df, title, output_path=None):
    for station_id, g_station in df.groupby("station_id"):
        plt.figure(figsize=(14, 5))
        obs_series = g_station[["valid_time", "obs_wind_speed"]].drop_duplicates().sort_values("valid_time")
        plt.plot(obs_series["valid_time"], obs_series["obs_wind_speed"], label="OBS", linewidth=2)
        for model, g in g_station.groupby("model"):
            g = g.sort_values("valid_time")
            plt.plot(g["valid_time"], g["forecast_wind_speed"], label=model, alpha=0.85)
        plt.title(f"{title} | {station_id}")
        plt.xlabel("Valid time")
        plt.ylabel("10m wind speed / station wind speed (m/s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            out = Path(output_path) / f"{title.replace(' ', '_').replace('/', '_')}_{station_id}_timeseries.png"
            plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.show()

def plot_metric_bars(overall_metrics, title, output_path=None):
    for metric in ["RMSE", "MAE", "Bias"]:
        plt.figure(figsize=(8, 4))
        m = overall_metrics.sort_values(metric)
        plt.bar(m["model"], m[metric])
        plt.title(f"{title} | Overall {metric}")
        plt.ylabel(f"{metric} (m/s)")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        if output_path:
            out = Path(output_path) / f"{title.replace(' ', '_').replace('/', '_')}_overall_{metric}.png"
            plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.show()

# ## 5. Define aligned experiments

EXPERIMENTS = [
    {
        "name": "2016_FCN3_HRES_OBS_12h",
        "start": "2016-01-01",
        "end": "2016-05-12",
        "lead_hours": 12,
        "models": {
            "FCN3": FCN3_2016_ZARR,
            "HRES": HRES_0P25,
        },
    },
    {
        "name": "2018_FCN3_HRES_OBS_12h",
        "start": "2018-01-01",
        "end": "2018-05-20",
        "lead_hours": 12,
        "models": {
            "FCN3": FCN3_2018_ZARR,
            "HRES": HRES_0P25,
        },
    },
    {
        "name": "2022_FCN3_GRAPHCAST_PANGU_HRES_OBS_12h",
        "start": "2022-01-01",
        "end": "2022-01-28",
        "lead_hours": 12,
        "models": {
            "FCN3": FCN3_2022_ZARR,
            "GraphCast": GRAPHCAST_2022_ZARR,
            "Pangu": PANGU_0P25,
            "HRES": HRES_0P25,
        },
    },
]

for exp in EXPERIMENTS:
    print("\n", exp["name"])
    for model, path in exp["models"].items():
        print(f"  {model:10s}", path, "EXISTS" if Path(path).exists() else "MISSING")

# ## 6. Run all comparisons
# 
# This cell:
# 
# 1. downloads station metadata;
# 2. downloads observations for each period;
# 3. extracts model values at nearest grid point;
# 4. aligns all data by `station_id` and `valid_time`;
# 5. saves aligned CSV files, metrics CSV files, and plots.

stations_df = download_station_metadata(STATION_IDS)
display(stations_df)

all_aligned = []
all_metrics = []

for exp in EXPERIMENTS:
    name = exp["name"]
    start = exp["start"]
    end = exp["end"]
    lead_hours = exp["lead_hours"]

    print("\n" + "=" * 100)
    print("Running:", name)
    print("Period:", start, "to", end)
    print("Lead:", lead_hours, "h")

    obs_df = download_observations(STATION_IDS, start, end)
    print("OBS rows:", len(obs_df), "time:", obs_df["time"].min(), "to", obs_df["time"].max())

    model_frames = []
    for model_name, zarr_path in exp["models"].items():
        print("\nExtracting:", model_name)
        model_df = extract_model_at_stations(
            zarr_path=zarr_path,
            model_name=model_name,
            stations_df=stations_df,
            start_date=start,
            end_date=end,
            lead_hours=lead_hours,
        )
        print(model_name, "rows:", len(model_df), "time:", model_df["valid_time"].min(), "to", model_df["valid_time"].max())
        model_frames.append(model_df)

    model_df = pd.concat(model_frames, ignore_index=True)
    aligned = align_with_obs(model_df, obs_df)
    aligned["experiment"] = name

    by_station, overall = compute_metrics(aligned)
    by_station["experiment"] = name
    overall["experiment"] = name

    print("\nOverall metrics:")
    display(overall)

    print("\nBy-station metrics:")
    display(by_station)

    exp_dir = OUTPUT_DIR / name
    exp_dir.mkdir(exist_ok=True, parents=True)

    aligned.to_csv(exp_dir / "aligned_forecast_obs.csv", index=False)
    overall.to_csv(exp_dir / "metrics_overall.csv", index=False)
    by_station.to_csv(exp_dir / "metrics_by_station.csv", index=False)

    plot_timeseries(aligned, name, output_path=exp_dir)
    plot_metric_bars(overall, name, output_path=exp_dir)

    all_aligned.append(aligned)
    all_metrics.append(overall)

all_aligned_df = pd.concat(all_aligned, ignore_index=True)
all_metrics_df = pd.concat(all_metrics, ignore_index=True)

all_aligned_df.to_csv(OUTPUT_DIR / "all_aligned_forecast_obs.csv", index=False)
all_metrics_df.to_csv(OUTPUT_DIR / "all_metrics_overall.csv", index=False)

print("\nSaved combined outputs to:", OUTPUT_DIR.resolve())
display(all_metrics_df)

# ## 7. Optional: compare only common valid times across all models inside each experiment
# 
# The default alignment above compares each model against OBS whenever that model and OBS overlap.
# This optional cell restricts each experiment to timestamps where **all models** are available at the same station and valid time.

def restrict_to_common_model_times(aligned):
    frames = []
    for (experiment, station_id), g in aligned.groupby(["experiment", "station_id"]):
        models = sorted(g["model"].unique())
        counts = g.groupby("valid_time")["model"].nunique().reset_index(name="n_models")
        common_times = counts.loc[counts["n_models"] == len(models), "valid_time"]
        frames.append(g[g["valid_time"].isin(common_times)].copy())
    return pd.concat(frames, ignore_index=True)

common_aligned_df = restrict_to_common_model_times(all_aligned_df)
common_metrics_rows = []

for exp_name, g in common_aligned_df.groupby("experiment"):
    _, overall = compute_metrics(g)
    overall["experiment"] = exp_name
    common_metrics_rows.append(overall)

common_metrics_df = pd.concat(common_metrics_rows, ignore_index=True)

common_aligned_df.to_csv(OUTPUT_DIR / "all_aligned_common_model_times.csv", index=False)
common_metrics_df.to_csv(OUTPUT_DIR / "all_metrics_common_model_times.csv", index=False)

display(common_metrics_df)

# ## 8. Notes for interpretation
# 
# Recommended reporting structure:
# 
# 1. **2016**: FCN3 vs HRES vs OBS, +12h, within-training/reference year.
# 2. **2018**: FCN3 vs HRES vs OBS, +12h, out-of-training/reference year depending on your experiment design.
# 3. **2022**: FCN3 vs GraphCast vs Pangu vs HRES vs OBS, +12h, fully aligned short-period comparison.
# 
# Be careful that station observations are point measurements, while model values are nearest-grid-cell values. The comparison therefore includes representativeness error, especially in complex terrain / coastal regions around Tromsø.

