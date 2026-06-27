# Case Study — How to Run Jobs

Northern Norway wind forecast evaluation:
**FCN3** and **GraphCast** vs HRES, ERA5 and station observations,
for years 2016, 2017 (in-training) and 2020–2022 (out-of-training).

---

## Prerequisites

### Data that must already exist

| What | Path |
|---|---|
| ERA5 6-hourly NetCDF | `/cluster/work/projects/nn8106k/siyan/era5_6h/` |
| HRES WeatherBench2 zarr | `/cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts/hres/0p25/2016-2022-0012-1440x721.zarr` |
| Station observations CSV | `experiments/outputs/wind_obs/SN*_{year}.csv` |
| GraphCast model weights | `/cluster/work/projects/nn8106k/siyan/earth2studio_cache/models/graphcast_local` |

ERA5 file naming conventions:

| Type | Pattern |
|---|---|
| Single-level | `era5_single_{year}.nc` (full year) |
| Pressure-level | `era5_pressure_{year}_{MM}.nc` (one per month) |
| Vertical velocity (w) | `era5_w_{year}_{MM}.nc` (one per month, GraphCast only) |
| GraphCast init buffer | `era5_{single,pressure,w}_{year}_dec31.nc` or `era5_w_{year}_12.nc` |

Check ERA5 coverage before submitting:

```bash
ls /cluster/work/projects/nn8106k/siyan/era5_6h/era5_single_{2016,2017,2020,2021,2022}.nc
ls /cluster/work/projects/nn8106k/siyan/era5_6h/era5_pressure_{2016,2017,2020,2021,2022}_01.nc
# GraphCast w data (needed for all case-study years + Dec of previous year for init):
ls /cluster/work/projects/nn8106k/siyan/era5_6h/era5_w_{2015,2016,2017,2020,2021,2022}_12.nc
```

### Python environments

| Model | Environment |
|---|---|
| FCN3 | `/cluster/projects/nn8106k/siyan/envs/earth2studio` |
| GraphCast | `/cluster/projects/nn8106k/siyan/envs/earth2studio-graphcast` |
| Analysis only | CPU — `pip install --user` inside job |

---

## Step 1 — Run FCN3 forecasts (all 5 years)

```bash
cd /cluster/home/siyan/github/WF-experiments
sbatch experiments/jobs/run_case_study_fcn3.slurm
```

This submits an array job (5 tasks, one per year, running sequentially `%1`).
Each task runs the full year of weekly FCN3 forecasts and saves compact station zarrs to:

```
/cluster/work/projects/nn8106k/siyan/WF-experiments/case-study/forecasts/FCN3/{year}/
```

Estimated time per year: ~6 h on a single GPU.

To run a single year only:

```bash
sbatch --array=0 experiments/jobs/run_case_study_fcn3.slurm   # 2016
sbatch --array=1 experiments/jobs/run_case_study_fcn3.slurm   # 2017
sbatch --array=2 experiments/jobs/run_case_study_fcn3.slurm   # 2020
sbatch --array=3 experiments/jobs/run_case_study_fcn3.slurm   # 2021
sbatch --array=4 experiments/jobs/run_case_study_fcn3.slurm   # 2022
```

---

## Step 2 — Run GraphCast forecasts (all 5 years)

```bash
sbatch experiments/jobs/run_case_study_graphcast.slurm
```

Same array structure (5 tasks). GraphCast uses 6-hourly inits so it produces
~4× more forecast files per year; each task runs ~20 h.

> **Memory note:** GraphCast (JAX) pre-allocates ~90% of GPU memory (~85 GiB on a 95 GiB GPU).
> To avoid OOM, forecasts are processed **one init time at a time** inside the loop.
> Each week's results are concatenated after all inits complete.

Output path:

```
/cluster/work/projects/nn8106k/siyan/WF-experiments/case-study/forecasts/GraphCast/{year}/
```

---

## Step 3 — Run analysis (metrics + figures)

Once forecasts are saved, run analysis — no GPU required, but the cluster only has the `accel` partition so a GPU is still allocated (unused).

```bash
# FCN3
sbatch --export=MODEL=fcn3 experiments/jobs/run_case_study_analysis_only.slurm

# GraphCast
sbatch --export=MODEL=graphcast experiments/jobs/run_case_study_analysis_only.slurm
```

Outputs:

```
/cluster/work/projects/nn8106k/siyan/WF-experiments/case-study/
├── figures/
│   ├── station_map_grid.png
│   └── metrics_by_year_obs.png
└── metrics_summary.csv
```

---

## Re-running and overwrite

By default, existing weekly zarr chunks are skipped. To force a full re-run:

```bash
# Edit the slurm script and add --overwrite to the python command, or run directly:
python experiments/scripts/nldl-src/case_study_analysis.py \
    --model fcn3 --year 2016 --overwrite
```

---

## Monitoring jobs

```bash
# List your running/pending jobs
squeue -u $USER

# Watch a specific job
squeue -j <JOBID>

# Check array task status
squeue -j <ARRAY_JOBID>

# View live log for a running task
tail -f experiments/jobs/logs/case_study_fcn3_<JOBID>_<TASKID>.out
```

---

## Output structure

```
/cluster/work/projects/nn8106k/siyan/WF-experiments/case-study/
├── forecasts/
│   ├── FCN3/
│   │   ├── 2016/   fcn3_20160101_20160107.zarr  ...
│   │   ├── 2017/
│   │   ├── 2020/
│   │   ├── 2021/
│   │   └── 2022/
│   └── GraphCast/
│       └── {same year structure}
├── figures/
│   ├── station_map_grid.png
│   └── metrics_by_year_obs.png
└── metrics_summary.csv
```

Each weekly zarr contains dims `(station, time, lead_time)` with 10 m wind speed
for all 12 evaluation stations.

---

## Module files

| Script | Purpose |
|---|---|
| `case_study_analysis.py` | CLI entry point |
| `config.py` | stations, paths, year lists |
| `local_era5.py` | reads ERA5 from local NetCDF |
| `forecasting.py` | Steps 1–3: data source, model load, run |
| `data_loading.py` | load obs / HRES / forecast zarrs |
| `metrics.py` | RMSE, MSE, Bias, RSE |
| `analysis.py` | Step 4 orchestration |
| `visualization.py` | station map, metric bar charts |
