#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=fcn3_cds_station
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/jobs/.out/%x_%j.out
#SBATCH --error=experiments/jobs/.err/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Starting FCN3 CDS ERA5 station wind prediction"
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
mkdir -p experiments/jobs/.out experiments/jobs/.err

echo "Python:"
which python
python --version

echo "CUDA:"
which nvcc
nvcc --version

echo "Checking CDS API config:"
if [ -f "$HOME/.cdsapirc" ]; then
    echo "~/.cdsapirc found"
else
    echo "WARNING: ~/.cdsapirc not found"
fi

echo "Running CDS ERA5 station wind script..."

python experiments/scripts/run_fcn3_cds_era5_station_wind_2009_2012.py \
  --start "2009-01-01 00:00" \
  --end "2012-12-31 18:00" \
  --nsteps 1 \
  --output-dir experiments/outputs

echo "============================================================"
echo "Finished FCN3 CDS ERA5 station wind prediction"
echo "Time: $(date)"
echo "============================================================"