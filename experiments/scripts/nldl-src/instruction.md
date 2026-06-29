# Case Study — How to Run Jobs

Northern Norway wind forecast evaluation:
**FCN3** and **GraphCast** vs HRES, ERA5 and station observations,
for years 2016, 2017 (in-training) and 2018–2022, 2024 (out-of-training).

Metrics and figures are produced in the notebook:
`experiments/notebooks/fcn3_case_study_analysis.ipynb`

---

## Prerequisites

### Data that must already exist

| What | Path |
|---|---|
| ERA5 6-hourly NetCDF | `/cluster/work/projects/nn8106k/siyan/era5_6h/` |
| HRES forecast zarr (0/12 UTC init) | `/cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts/hres/0p25/2016-2022-0012-1440x721.zarr` |
| Station observations CSV | `experiments/outputs/wind_obs/SN*_{year}.csv` |
| GraphCast model weights | `/cluster/work/projects/nn8106k/siyan/earth2studio_cache/models/graphcast_local` |

Download HRES forecast zarr if missing:

```bash
sbatch experiments/jobs/download_hres_forecast_wb2.slurm
```

ERA5 file naming conventions:

| Type | Pattern |
|---|---|
| Single-level | `era5_single_{year}.nc` (full year) |
| Pressure-level | `era5_pressure_{year}_{MM}.nc` (one per month) |
| Vertical velocity (w) | `era5_w_{year}_{MM}.nc` (one per month, GraphCast only) |
| GraphCast init buffer | `era5_{single,pressure,w}_{year}_dec31.nc` or `era5_w_{year}_12.nc` |

Check ERA5 coverage before submitting:

```bash
ls /cluster/work/projects/nn8106k/siyan/era5_6h/era5_single_{2016,2017,2018,2019,2020,2021,2022,2024}.nc
ls /cluster/work/projects/nn8106k/siyan/era5_6h/era5_pressure_{2016,2017,2018,2019,2020,2021,2022,2024}_01.nc
# GraphCast w data (needed for all case-study years + Dec of previous year for init):
ls /cluster/work/projects/nn8106k/siyan/era5_6h/era5_w_{2015,2016,2017,2018,2019,2020,2021,2022,2023}_12.nc
```

### Python environments

| Model | Environment |
|---|---|
| FCN3 | `/cluster/projects/nn8106k/siyan/envs/earth2studio` |
| GraphCast | `/cluster/projects/nn8106k/siyan/envs/earth2studio-graphcast` |
| Notebook (metrics & figures) | `/cluster/projects/nn8106k/siyan/envs/e2s-notebook` |

---

## Step 1 — Run FCN3 forecasts

```bash
cd /cluster/home/siyan/github/WF-experiments
sbatch experiments/jobs/run_case_study_fcn3.slurm
```

Array job, one task per year. Each task runs the full year of weekly FCN3 forecasts
and saves compact station zarrs to:

```
/cluster/work/projects/nn8106k/siyan/WF-experiments/case-study/forecasts/FCN3/{year}/
```

Estimated time per year: ~6 h on a single GPU.

To run a single year:

```bash
sbatch --array=0 experiments/jobs/run_case_study_fcn3.slurm   # year index 0
```

Check the array index → year mapping at the top of the SLURM script.

---

## Step 2 — Run GraphCast forecasts

```bash
sbatch experiments/jobs/run_case_study_graphcast.slurm
```

Same array structure. GraphCast uses 6-hourly inits → ~4× more files per year; each task ~20 h.

> **Memory note:** GraphCast (JAX) pre-allocates ~90% of GPU memory (~85 GiB on a 95 GiB GPU).
> Forecasts are processed one init time at a time to avoid OOM.

Output path:

```
/cluster/work/projects/nn8106k/siyan/WF-experiments/case-study/forecasts/GraphCast/{year}/
```

---

## Step 3 — Compute metrics and generate figures (notebook)

Open `experiments/notebooks/fcn3_case_study_analysis.ipynb` and run all cells.

The notebook:
- Loads FCN3 forecast zarrs, HRES zarr, ERA5 NetCDF, and station obs CSV
- Computes RMSE, MAE, nRMSE, Bias per station × lead time (+6/+12/+18/+24 h)
- Caches results to CSV — rerunning reloads from cache unless `FORCE_RECOMPUTE = True`
- Produces heatmaps, annual bar charts, summary tables, and skill score comparison

Cached CSV files:

```
/cluster/work/projects/nn8106k/siyan/WF-experiments/case-study/
├── metrics_fcn3_vs_era5.csv
├── metrics_fcn3_vs_obs.csv
├── metrics_hres_vs_era5.csv
└── metrics_hres_vs_obs.csv
```

---

## Re-running and overwrite

Forecast zarrs: existing weekly chunks are skipped by default. To force re-run:

```bash
python experiments/scripts/nldl-src/case_study_analysis.py \
    --model fcn3 --year 2016 --overwrite
```

Metrics: set `FORCE_RECOMPUTE = True` in the notebook config cell and rerun.

---

## Monitoring jobs

```bash
squeue -u $USER                        # list running/pending jobs
squeue -j <JOBID>                      # specific job
tail -f experiments/jobs/logs/<name>_<JOBID>.out   # live log
```

---

## Output structure

```
/cluster/work/projects/nn8106k/siyan/WF-experiments/case-study/
├── forecasts/
│   ├── FCN3/
│   │   ├── 2016/   fcn3_20160101_20160107.zarr  ...
│   │   ├── 2017/
│   │   ├── 2018/
│   │   ├── 2019/
│   │   ├── 2020/
│   │   ├── 2021/
│   │   ├── 2022/
│   │   └── 2024/
│   └── GraphCast/
│       └── {same year structure}
├── figures/
├── metrics_fcn3_vs_era5.csv
├── metrics_fcn3_vs_obs.csv
├── metrics_hres_vs_era5.csv
└── metrics_hres_vs_obs.csv
```

Each weekly zarr contains dims `(station, time, lead_time)` with 10 m wind speed
for all 12 evaluation stations.

---

## Module files

| Script | Purpose |
|---|---|
| `case_study_analysis.py` | CLI entry point (Steps 1–2: forecast generation) |
| `config.py` | stations, paths, year lists |
| `local_era5.py` | reads ERA5 from local NetCDF |
| `forecasting.py` | data source, model load, run forecasts |
