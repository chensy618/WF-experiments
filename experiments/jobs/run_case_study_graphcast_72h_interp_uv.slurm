#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=case_study_gc_72h_interp_uv
#SBATCH --time=40:00:00
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/nldl/%x_%A_%a.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/nldl/%x_%A_%a.err

# Same 2-day/3-day (+72h) GraphCast case study as
# run_case_study_graphcast_72h_interp.slurm (bilinear-interpolation spatial
# matching, --extraction interp), but adds --save-uv so u10m/v10m are also
# saved as extra data variables alongside the scalar wind_speed_10m (which is
# still saved, unchanged). Writes to forecasts/GraphCast_72h_interp_uv/, so it
# never overwrites the existing forecasts/GraphCast_72h_interp/ data used by
# the current notebooks.
#
# All other settings (nsteps, output dir suffix, init cadence) come from
# config.py's NSTEPS_LONG/LONG_OUT_TAG/LONG_DAILY_ONLY via --preset long —
# edit config.py, not this file.
# LONG_DAILY_ONLY forces GraphCast down to daily 00 UTC init (same cadence as
# FCN3) instead of its usual 6-hourly, to keep the +72h comparison fair and
# the compute tractable.
#
# Array index → year mapping: 0→2016, 1→2017, 2→2018, 3→2019, 4→2020, 5→2021, 6→2022
YEARS=(2016 2017 2018 2019 2020 2021 2022)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

set -euo pipefail

echo "============================================================"
echo "Case study — GraphCast 2-day/3-day forecast (--preset long, --extraction interp, --save-uv)"
echo "Year: ${YEAR}  (task ${SLURM_ARRAY_TASK_ID} / job ${SLURM_ARRAY_JOB_ID})"
echo "Node: $(hostname)  |  $(date)"
echo "============================================================"

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/13.0.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio-graphcast/bin/activate
source ~/.ngc_env

cd /cluster/home/siyan/github/WF-experiments

export CUDA_HOME=${EBROOTCUDA:-$(dirname $(dirname $(which nvcc)))}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARTH2STUDIO_CACHE=/cluster/work/projects/nn8106k/siyan/earth2studio_cache
export EARTH2STUDIO_DATA_CACHE=$EARTH2STUDIO_CACHE/data
export EARTH2STUDIO_MODEL_CACHE=$EARTH2STUDIO_CACHE/models
export GC_PACKAGE_PATH=/cluster/work/projects/nn8106k/siyan/earth2studio_cache/models/graphcast_local

mkdir -p "$EARTH2STUDIO_DATA_CACHE" "$EARTH2STUDIO_MODEL_CACHE"
mkdir -p experiments/jobs/logs/nldl

python -u experiments/scripts/nldl-src/case_study_analysis.py \
    --model graphcast \
    --year "${YEAR}" \
    --preset long \
    --extraction interp \
    --save-uv \
    --package-path "${GC_PACKAGE_PATH}"

echo "============================================================"
echo "Finished GraphCast 2-day/3-day case study (interp, save-uv) — year ${YEAR}"
echo "$(date)"
echo "============================================================"
