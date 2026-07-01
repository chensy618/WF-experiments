#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=hres_forecast_wb2
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.err

set -euo pipefail

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo ""
echo "============================================================"
echo "HRES forecast zarr download: 2016-2022-0012-1440x721.zarr"
echo "Source: gs://weatherbench2/datasets/hres_forecasts/1440x721/"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================================"

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/13.0.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio/bin/activate

# NOTE: compute nodes on this cluster block outbound HTTPS.
# This SLURM job will fail. Run the download on the login node instead — see below.
# If a proxy is available, set: export https_proxy=http://<proxy_ip>:<port>

cd /cluster/home/siyan/github/WF-experiments

python -u experiments/scripts/download_hres_forecast_wb2.py

echo "============================================================"
echo "Finished HRES forecast download"
echo "Time: $(date)"
echo "============================================================"
