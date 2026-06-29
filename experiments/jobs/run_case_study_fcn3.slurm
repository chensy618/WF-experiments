#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=case_study_fcn3
#SBATCH --time=08:00:00
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/nldl/%x_%A_%a.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/nldl/%x_%A_%a.err

# Array index → year mapping: 0→2018, 1→2019, 2→2023, 3→2024
YEARS=(2018 2019 2023 2024)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

set -euo pipefail

echo "============================================================"
echo "Case study — FCN3 forecast"
echo "Year: ${YEAR}  (task ${SLURM_ARRAY_TASK_ID} / job ${SLURM_ARRAY_JOB_ID})"
echo "Node: $(hostname)  |  $(date)"
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

export EARTH2STUDIO_CACHE=/cluster/work/projects/nn8106k/siyan/earth2studio_cache
export EARTH2STUDIO_DATA_CACHE=$EARTH2STUDIO_CACHE/data
export EARTH2STUDIO_MODEL_CACHE=$EARTH2STUDIO_CACHE/models

mkdir -p "$EARTH2STUDIO_DATA_CACHE" "$EARTH2STUDIO_MODEL_CACHE"
mkdir -p experiments/jobs/logs/nldl

python -u experiments/scripts/nldl-src/case_study_analysis.py \
    --model fcn3 \
    --year "${YEAR}"

echo "============================================================"
echo "Finished FCN3 case study — year ${YEAR}"
echo "$(date)"
echo "============================================================"
