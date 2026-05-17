from pathlib import Path

import torch

from earth2studio.models.px import FCN3
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic


def main():
    print("============================================================")
    print("FCN3 small test")
    print("============================================================")

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    output_dir = Path("experiments/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = output_dir / "fcn3_small.zarr"

    print("Output path:", zarr_path)

    print("Loading FCN3 package...")
    package = FCN3.load_default_package()

    print("Loading FCN3 model...")
    model = FCN3.load_model(package)

    print("Loading GFS data source...")
    data = GFS(source="aws", cache=True, verbose=True)

    print("Creating Zarr backend...")
    io = ZarrBackend(file_name=str(zarr_path))

    print("Running deterministic forecast...")
    print("Initial time: 2024-01-01")
    print("Forecast steps: 1")

    io = deterministic(
        ["2024-01-01"],
        1,
        model,
        data,
        io,
    )

    print("============================================================")
    print("FCN3 small test finished successfully")
    print("Saved output to:", zarr_path)
    print("============================================================")


if __name__ == "__main__":
    main()