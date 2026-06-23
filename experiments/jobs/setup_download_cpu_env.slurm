#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=small
#SBATCH --job-name=setup_download_env
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.err

set -euo pipefail

echo "Setting up download-cpu venv on $(hostname) at $(date)"

module --force purge
module load NRIS/CPU
module load Python/3.11.5-GCCcore-13.2.0

VENV=/cluster/projects/nn8106k/siyan/envs/download-cpu

python -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip
pip install cdsapi gcsfs zarr aiohttp requests

echo "Done. Packages installed:"
pip list | grep -E "cdsapi|gcsfs|zarr|aiohttp|requests"
echo "Finished at $(date)"
