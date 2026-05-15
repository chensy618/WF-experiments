#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=normal
#SBATCH --job-name=combine_fcn3_weekly
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=experiments/jobs/.out/%x_%j.out
#SBATCH --error=experiments/jobs/.err/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Combining weekly FCN3 station outputs"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

module --force purge
module load NRIS/Login
module load Python/3.11.5-GCCcore-13.2.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio/bin/activate

cd /cluster/home/siyan/github/WF-experiments

mkdir -p /cluster/projects/nn8106k/siyan/WF-experiments/outputs/fcn3_station_combined
mkdir -p experiments/jobs/.out experiments/jobs/.err

python -u experiments/scripts/combine_fcn3_station_weekly.py \
  --input-dir /cluster/projects/nn8106k/siyan/WF-experiments/outputs/fcn3_station_weekly \
  --output-dir /cluster/projects/nn8106k/siyan/WF-experiments/outputs/fcn3_station_combined \
  --output-name fcn3_station_wind_2009_2012.zarr \
  --start "2009-01-01 00:00" \
  --end "2012-12-31 18:00"

echo "============================================================"
echo "Finished combining weekly FCN3 station outputs"
echo "Time: $(date)"
echo "============================================================"