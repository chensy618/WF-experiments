#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=era5_6h_download
#SBATCH --time=120:00:00
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "ERA5 6-hourly download: 2023"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/13.0.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio/bin/activate

cd /cluster/home/siyan/github/WF-experiments

OUTPUT_DIR=/cluster/work/projects/nn8106k/siyan/era5_6h
mkdir -p "$OUTPUT_DIR"

python -u experiments/scripts/download_era5_6h.py \
  --years 2023 \
  --output-dir "$OUTPUT_DIR" \
  --prev-dec31

echo "============================================================"
echo "Finished ERA5 2023 download"
echo "Time: $(date)"
echo "============================================================"
