#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=graphcast_cds_daytest
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-7%1
#SBATCH --output=experiments/jobs/.out/%x_%A_%a.out
#SBATCH --error=experiments/jobs/.err/%x_%A_%a.err

set -euo pipefail

echo "============================================================"
echo "Running GraphCastOperational + CDS ERA5 daily weekly-test"
echo "Job ID: ${SLURM_JOB_ID}"
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

export EARTH2STUDIO_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_graphcast_cache
export EARTH2STUDIO_DATA_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_graphcast_cache/data
export EARTH2STUDIO_MODEL_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_graphcast_cache/models

mkdir -p "$EARTH2STUDIO_DATA_CACHE"
mkdir -p "$EARTH2STUDIO_MODEL_CACHE"
mkdir -p experiments/outputs/graphcast_operational_cds_era5_daily
mkdir -p experiments/jobs/.out experiments/jobs/.err

# Public Google Cloud Storage access for GraphCast model package / weights
export GCSFS_TOKEN=anon
export GOOGLE_APPLICATION_CREDENTIALS=""
export CLOUDSDK_CONFIG=/tmp/nonexistent-gcloud-config-${SLURM_JOB_ID}

echo "Checking CDS API config:"
ls -la ~/.cdsapirc || true

echo "Proxy environment variables:"
env | grep -i proxy || true

python experiments/scripts/run_graphcast_operational_cds_era5_daily.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --base-date "2009-01-01T00:00:00" \
    --nsteps 4 \
    --output-root experiments/outputs/graphcast_operational_cds_era5_daily

echo "============================================================"
echo "Finished GraphCastOperational + CDS ERA5 daily weekly-test task"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Time: $(date)"
echo "============================================================"