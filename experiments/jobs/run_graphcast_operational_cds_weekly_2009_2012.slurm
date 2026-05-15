#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=graphcast_cds_weekly
#SBATCH --time=24:00:00
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-208%1
#SBATCH --output=experiments/jobs/.out/%x_%A_%a.out
#SBATCH --error=experiments/jobs/.err/%x_%A_%a.err

set -euo pipefail

echo "============================================================"
echo "Running GraphCastOperational + CDS ERA5 station wind weekly job"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/13.0.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio-graphcast/bin/activate

cd /cluster/home/siyan/github/WF-experiments

export CUDA_HOME=${EBROOTCUDA:-$(dirname $(dirname $(which nvcc)))}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export EARTH2STUDIO_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_graphcast_cache
export EARTH2STUDIO_DATA_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_graphcast_cache/data
export EARTH2STUDIO_MODEL_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_graphcast_cache/models

mkdir -p "$EARTH2STUDIO_DATA_CACHE"
mkdir -p "$EARTH2STUDIO_MODEL_CACHE"
mkdir -p /cluster/projects/nn8106k/siyan/WF-experiments/outputs/graphcast_operational_cds_weekly
mkdir -p experiments/jobs/.out experiments/jobs/.err

# Public Google Cloud Storage access for GraphCast model package / weights
export GCSFS_TOKEN=anon
export GOOGLE_APPLICATION_CREDENTIALS=""
export CLOUDSDK_CONFIG=/tmp/nonexistent-gcloud-config-${SLURM_JOB_ID}

START_DATE="2009-01-01"
END_DATE="2012-12-31"

WEEK_START=$(python - <<PY
from datetime import datetime, timedelta
start = datetime.strptime("${START_DATE}", "%Y-%m-%d")
d = start + timedelta(days=7 * int("${SLURM_ARRAY_TASK_ID}"))
print(d.strftime("%Y-%m-%d"))
PY
)

echo "Resolved week start:"
echo "${WEEK_START}"

echo "Checking CDS API config:"
ls -la ~/.cdsapirc || true

echo "Proxy environment variables:"
env | grep -i proxy || true

python -u experiments/scripts/run_graphcast_operational_cds_station_wind_one_week.py \
  --start-date "${WEEK_START}" \
  --days 7 \
  --end-date "${END_DATE}" \
  --nsteps 4 \
  --output-dir /cluster/projects/nn8106k/siyan/WF-experiments/outputs/graphcast_operational_cds_weekly

echo "============================================================"
echo "Finished weekly GraphCastOperational + CDS ERA5 job"
echo "Week start: ${WEEK_START}"
echo "Time: $(date)"
echo "============================================================"