#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=era5_w_download
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=0-1
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%A_%a.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%A_%a.err

# Array index → year mapping: 0→2021, 1→2022
YEARS=(2021 2022)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

set -euo pipefail

echo "============================================================"
echo "ERA5 vertical velocity (w) download — year ${YEAR}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)  |  $(date)"
echo "============================================================"

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio-graphcast/bin/activate
module load CUDA/13.0.0

export CUDA_HOME=${EBROOTCUDA:-$(dirname $(dirname $(which nvcc)))}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

cd /cluster/home/siyan/github/WF-experiments

OUTPUT_DIR=/cluster/work/projects/nn8106k/siyan/era5_6h
mkdir -p "$OUTPUT_DIR"

python -u experiments/scripts/download_era5_6h.py \
    --years "${YEAR}" \
    --output-dir "$OUTPUT_DIR" \
    --w-only

echo "============================================================"
echo "Finished ERA5 vertical velocity download — year ${YEAR}"
echo "$(date)"
echo "============================================================"
