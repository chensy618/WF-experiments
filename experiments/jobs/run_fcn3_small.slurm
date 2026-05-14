#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=fcn3_small
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/jobs/.out/%x_%j.out
#SBATCH --error=experiments/jobs/.err/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Starting FCN3 small test"
echo "Job ID: ${SLURM_JOB_ID}"
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

export EARTH2STUDIO_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_cache
export EARTH2STUDIO_DATA_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_cache/data
export EARTH2STUDIO_MODEL_CACHE=/cluster/projects/nn8106k/siyan/earth2studio_cache/models

mkdir -p "$EARTH2STUDIO_DATA_CACHE"
mkdir -p "$EARTH2STUDIO_MODEL_CACHE"
mkdir -p experiments/outputs

echo "Python:"
which python
python --version

echo "CUDA check:"
which nvcc
nvcc --version

echo "Running script..."
python experiments/scripts/run_fcn3_small.py

echo "============================================================"
echo "Finished FCN3 small test"
echo "Time: $(date)"
echo "============================================================"