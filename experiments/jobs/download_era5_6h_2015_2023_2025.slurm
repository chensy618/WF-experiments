#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=small
#SBATCH --job-name=era5_6h_download
#SBATCH --time=120:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.err

set -euo pipefail

echo "Starting job ${SLURM_JOB_ID} on $(hostname) on Olivia at $(date)"
echo ""
echo "============================================================"
echo "ERA5 6-hourly download: 2015, 2023, 2025"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

module --force purge
module load NRIS/CPU
module load Python/3.11.5-GCCcore-13.2.0

source /cluster/projects/nn8106k/siyan/envs/download-cpu/bin/activate

cd /cluster/home/siyan/github/WF-experiments

OUTPUT_DIR=/cluster/work/projects/nn8106k/siyan/era5_6h
mkdir -p "$OUTPUT_DIR"

python -u experiments/scripts/download_era5_6h.py \
  --years 2015 2023 2025 \
  --output-dir "$OUTPUT_DIR" \
  --prev-dec31

echo "============================================================"
echo "Finished ERA5 download"
echo "Time: $(date)"
echo "============================================================"
