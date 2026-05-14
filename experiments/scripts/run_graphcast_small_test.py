from pathlib import Path
import os
import sys
import aiohttp
import torch
import jax

# ---------------------------------------------------------------------
# Make aiohttp/gcsfs respect Olivia proxy environment variables.
# Needed for GraphCast files from Google Cloud Storage.
# ---------------------------------------------------------------------
_original_client_session = aiohttp.ClientSession

def _proxy_aware_client_session(*args, **kwargs):
    kwargs["trust_env"] = True
    return _original_client_session(*args, **kwargs)

aiohttp.ClientSession = _proxy_aware_client_session


from earth2studio.models.px import GraphCastSmall
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic


def main():
    print("============================================================")
    print("GraphCastSmall one-step inference test")
    print("============================================================")

    print("Python:", sys.executable)
    print("Torch:", torch.__version__)
    print("Torch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Torch GPU:", torch.cuda.get_device_name(0))

    print("JAX:", jax.__version__)
    print("JAX devices:", jax.devices())

    print("Proxy env:")
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        print(f"  {key} =", os.environ.get(key))

    print("GCSFS_TOKEN:", os.environ.get("GCSFS_TOKEN"))

    output_dir = Path("experiments/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = output_dir / "graphcast_small_test_20220101.zarr"

    if zarr_path.exists():
        raise FileExistsError(f"Output already exists: {zarr_path}")

    print("\nLoading GraphCastSmall package...")
    package = GraphCastSmall.load_default_package()
    print("Package loaded.")

    print("\nLoading GraphCastSmall model...")
    model = GraphCastSmall.load_model(package)
    print("Model loaded:", type(model))

    print("\nPreparing GFS data source...")
    data = GFS(source="aws", cache=True, verbose=True)

    print("\nPreparing Zarr output backend...")
    io = ZarrBackend(file_name=str(zarr_path))

    print("\nRunning deterministic forecast...")
    print("Initial time: 2022-01-01")
    print("Forecast steps: 1")

    io = deterministic(
        ["2022-01-01"],
        1,
        model,
        data,
        io,
    )

    print("\nSaved output to:", zarr_path)
    print("GraphCastSmall inference test finished successfully.")


if __name__ == "__main__":
    main()