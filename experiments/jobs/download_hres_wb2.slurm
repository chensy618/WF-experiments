#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=small
#SBATCH --job-name=hres_wb2_download
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.err

set -euo pipefail

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo ""
echo "============================================================"
echo "HRES WB2 zarr download: 2016-2022-6h-1440x721.zarr"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

module --force purge
module load NRIS/CPU
module load Python/3.11.5-GCCcore-13.2.0

pip install --user --quiet gcsfs

cd /cluster/home/siyan/github/WF-experiments

python -u experiments/scripts/download_hres_wb2.py

echo "============================================================"
echo "Finished HRES WB2 download"
echo "Time: $(date)"
echo "============================================================"
