#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=fcn3_cds_weekly
#SBATCH --time=06:00:00
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-208%1
#SBATCH --output=experiments/jobs/.out/%x_%A_%a.out
#SBATCH --error=experiments/jobs/.err/%x_%A_%a.err

set -euo pipefail

echo "============================================================"
echo "Running FCN3 + CDS station wind weekly job"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/13.0.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio/bin/activate
source ~/.ngc_env

cd /cluster/home/siyan/github/WF-experiments

export CUDA_HOME=${EBROOTCUDA:-$(dirname $(dirname $(which nvcc)))}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export EARTH2STUDIO_CACHE=/cluster/work/projects/nn8106k/siyan/earth2studio_cache
export EARTH2STUDIO_DATA_CACHE=/cluster/work/projects/nn8106k/siyan/earth2studio_cache/data
export EARTH2STUDIO_MODEL_CACHE=/cluster/work/projects/nn8106k/siyan/earth2studio_cache/models

OUTPUT_DIR=/cluster/work/projects/nn8106k/siyan/WF-experiments/outputs/fcn3_station_weekly

mkdir -p "$EARTH2STUDIO_DATA_CACHE"
mkdir -p "$EARTH2STUDIO_MODEL_CACHE"
mkdir -p "$OUTPUT_DIR"
mkdir -p experiments/jobs/.out experiments/jobs/.err

START_DATE="2009-01-01"

WEEK_START=$(python - <<PY
from datetime import datetime, timedelta

start = datetime.strptime("${START_DATE}", "%Y-%m-%d")
d = start + timedelta(days=7 * int("${SLURM_ARRAY_TASK_ID}"))
print(d.strftime("%Y-%m-%d"))
PY
)

WEEK_END=$(python - <<PY
from datetime import datetime, timedelta

start = datetime.strptime("${WEEK_START}", "%Y-%m-%d")
end = start + timedelta(days=6)
print(end.strftime("%Y-%m-%d"))
PY
)

SAFE_WEEK_START=$(echo "${WEEK_START}" | tr -d '-')
SAFE_WEEK_END=$(echo "${WEEK_END}" | tr -d '-')

EXPECTED_OUTPUT="${OUTPUT_DIR}/fcn3_station_wind_${SAFE_WEEK_START}_${SAFE_WEEK_END}.zarr"

echo "Resolved week range:"
echo "Week start: ${WEEK_START}"
echo "Week end:   ${WEEK_END}"
echo "Expected output:"
echo "${EXPECTED_OUTPUT}"

if [ -d "${EXPECTED_OUTPUT}" ]; then
    echo "============================================================"
    echo "Output already exists. Skipping this array task."
    echo "Existing output: ${EXPECTED_OUTPUT}"
    echo "Time: $(date)"
    echo "============================================================"
    exit 0
fi

echo "Output does not exist. Running this weekly forecast."

python -u experiments/scripts/run_fcn3_cds_station_wind_one_week.py \
  --start-date "${WEEK_START}" \
  --days 7 \
  --nsteps 1 \
  --output-dir "${OUTPUT_DIR}"

echo "============================================================"
echo "Finished weekly FCN3 + CDS station wind job"
echo "Week start: ${WEEK_START}"
echo "Week end:   ${WEEK_END}"
echo "Output:     ${EXPECTED_OUTPUT}"
echo "Time: $(date)"
echo "============================================================"