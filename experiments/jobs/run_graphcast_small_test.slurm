#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=graphcast_small_test
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/jobs/.out/%x_%j.out
#SBATCH --error=experiments/jobs/.err/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Running GraphCastSmall one-step inference test"
echo "Job ID: ${SLURM_JOB_ID}"
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
mkdir -p experiments/outputs
mkdir -p experiments/jobs/.out experiments/jobs/.err

# Public Google Cloud Storage access for DeepMind GraphCast files
export GCSFS_TOKEN=anon
export GOOGLE_APPLICATION_CREDENTIALS=""
export CLOUDSDK_CONFIG=/tmp/nonexistent-gcloud-config-${SLURM_JOB_ID}

echo "Proxy environment variables:"
env | grep -i proxy || true

python experiments/scripts/run_graphcast_small_test.py

echo "============================================================"
echo "Finished GraphCastSmall one-step inference test"
echo "Time: $(date)"
echo "============================================================"