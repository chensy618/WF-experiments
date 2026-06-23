#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=small
#SBATCH --job-name=wind_obs_northern_norway
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Wind obs download: 9 Northern Norway stations, 2016–2022"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

module --force purge
module load NRIS/CPU
module load Python/3.11.5-GCCcore-13.2.0

pip install --user --quiet pandas requests python-dotenv

cd /cluster/home/siyan/github/WF-experiments

python -u experiments/scripts/download_wind_obs_northern_norway.py

echo "============================================================"
echo "Finished wind obs download"
echo "Time: $(date)"
echo "============================================================"
