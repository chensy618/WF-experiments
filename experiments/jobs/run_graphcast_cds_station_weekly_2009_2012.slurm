#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=gc_station_weekly
#SBATCH --time=24:00:00
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-208%1
#SBATCH --output=experiments/jobs/.out/%x_%A_%a.out
#SBATCH --error=experiments/jobs/.err/%x_%A_%a.err

# GraphCast Operational – 6-hourly cycling station wind  (2009-01-01 → 2012-12-31)
#
# Aligned with FCN3 (run_fcn3_cds_station_weekly_2009_2012.slurm):
#   - 6-hourly initialization: 00:00, 06:00, 12:00, 18:00 UTC
#   - nsteps = 1  →  6 h lead time only
#   - Output: compact station zarr (no full global grid)
#
# Total GC passes per week: 28 inits × 1 step = 28
#   (same as old config: 7 inits × 4 steps = 28 — compute budget unchanged)

set -euo pipefail

echo "============================================================"
echo "GraphCast Operational + CDS ERA5 station wind (6-hourly)"
echo "Array job ID : ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node         : $(hostname)"
echo "Time         : $(date)"
echo "============================================================"

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/13.0.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio-graphcast/bin/activate

cd /cluster/home/siyan/github/WF-experiments

export CUDA_HOME=${EBROOTCUDA:-$(dirname $(dirname $(which nvcc)))}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

export EARTH2STUDIO_CACHE=/cluster/work/projects/nn8106k/siyan/earth2studio_graphcast_cache
export EARTH2STUDIO_DATA_CACHE=/cluster/work/projects/nn8106k/siyan/earth2studio_graphcast_cache/data
export EARTH2STUDIO_MODEL_CACHE=/cluster/work/projects/nn8106k/siyan/earth2studio_graphcast_cache/models

# Public Google Cloud Storage access for GraphCast model weights
export GCSFS_TOKEN=anon
export GOOGLE_APPLICATION_CREDENTIALS=""
export CLOUDSDK_CONFIG=/tmp/nonexistent-gcloud-config-${SLURM_JOB_ID}

OUTPUT_DIR=/cluster/work/projects/nn8106k/siyan/WF-experiments/outputs/graphcast_station_weekly

mkdir -p "${EARTH2STUDIO_DATA_CACHE}"
mkdir -p "${EARTH2STUDIO_MODEL_CACHE}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p experiments/jobs/.out experiments/jobs/.err

START_DATE="2009-01-01"
END_DATE="2012-12-31"
NSTEPS=1

WEEK_START=$(python - <<PY
from datetime import datetime, timedelta
start = datetime.strptime("${START_DATE}", "%Y-%m-%d")
week_start = start + timedelta(days=7 * int("${SLURM_ARRAY_TASK_ID}"))
print(week_start.strftime("%Y-%m-%d"))
PY
)

WEEK_END=$(python - <<PY
from datetime import datetime, timedelta
week_start = datetime.strptime("${WEEK_START}", "%Y-%m-%d")
end_date   = datetime.strptime("${END_DATE}", "%Y-%m-%d")
week_end   = min(week_start + timedelta(days=6), end_date)
print(week_end.strftime("%Y-%m-%d"))
PY
)

DAYS=$(python - <<PY
from datetime import datetime
ws = datetime.strptime("${WEEK_START}", "%Y-%m-%d")
we = datetime.strptime("${WEEK_END}",   "%Y-%m-%d")
print((we - ws).days + 1)
PY
)

SAFE_WEEK_START=$(echo "${WEEK_START}" | tr -d '-')
SAFE_WEEK_END=$(echo "${WEEK_END}"   | tr -d '-')

EXPECTED_OUTPUT="${OUTPUT_DIR}/graphcast_station_wind_${SAFE_WEEK_START}_${SAFE_WEEK_END}.zarr"
SUCCESS_MARKER="${EXPECTED_OUTPUT}/_SUCCESS"

echo "Week start   : ${WEEK_START}"
echo "Week end     : ${WEEK_END}"
echo "Days         : ${DAYS}"
echo "nsteps       : ${NSTEPS}"
echo "Expected out : ${EXPECTED_OUTPUT}"

if [ -f "${SUCCESS_MARKER}" ]; then
    echo "Output already complete. Skipping."
    exit 0
fi

if [ -d "${EXPECTED_OUTPUT}" ] && [ ! -f "${SUCCESS_MARKER}" ]; then
    echo "Incomplete output found (no _SUCCESS). Removing: ${EXPECTED_OUTPUT}"
    rm -rf "${EXPECTED_OUTPUT}"
fi

echo "Starting GraphCast station wind script..."

python -u experiments/scripts/run_graphcast_cds_station_wind_one_week.py \
  --start-date "${WEEK_START}" \
  --days       "${DAYS}"       \
  --nsteps     "${NSTEPS}"     \
  --output-dir "${OUTPUT_DIR}"

echo "============================================================"
echo "Finished GraphCast station wind job"
echo "Week : ${WEEK_START} → ${WEEK_END}"
echo "Out  : ${EXPECTED_OUTPUT}"
echo "Time : $(date)"
echo "============================================================"
