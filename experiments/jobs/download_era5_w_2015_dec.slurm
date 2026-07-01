#!/bin/bash
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --job-name=era5_w_2015_dec
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.out
#SBATCH --error=/cluster/home/siyan/github/WF-experiments/experiments/jobs/logs/%x_%j.err

set -euo pipefail

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0

source /cluster/projects/nn8106k/siyan/envs/earth2studio-graphcast/bin/activate

OUT=/cluster/work/projects/nn8106k/siyan/era5_6h/era5_w_2015_12.nc

if [ -f "$OUT" ]; then
    echo "Already exists: $OUT"
    exit 0
fi

python -u - <<'EOF'
import cdsapi
from pathlib import Path

out = Path("/cluster/work/projects/nn8106k/siyan/era5_6h/era5_w_2015_12.nc")
days = [f"{d:02d}" for d in range(1, 32)]
levels = ["50","100","150","200","250","300","400","500","600","700","850","925","1000"]

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "year": "2015",
        "month": "12",
        "day": days,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "variable": ["vertical_velocity"],
        "pressure_level": levels,
        "format": "netcdf",
        "grid": "0.25/0.25",
    },
    str(out),
)
print(f"Saved -> {out}")
EOF
