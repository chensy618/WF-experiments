"""One-init smoke test for the re-run: model weights download + ARCO ERA5 streaming +
72 h rollout + bilinear station extraction. Prints timing and station winds.

    python smoke_test_arco.py --model fcn3|graphcast
"""
import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import _EXTRACTION_FUNCS, load_data_source, load_model  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["fcn3", "graphcast"], required=True)
p.add_argument("--raw", default="/cluster/work/projects/nn8106k/siyan/smoke_raw")
a = p.parse_args()

from earth2studio.io import ZarrBackend  # noqa: E402
from earth2studio.run import deterministic  # noqa: E402

t0 = time.time()
model, label = load_model(a.model)
print(f"weights loaded in {time.time() - t0:.0f}s", flush=True)
data = load_data_source(source="arco")
raw = Path(a.raw) / f"{label}_raw.zarr"
raw.parent.mkdir(parents=True, exist_ok=True)
if raw.exists():
    shutil.rmtree(raw)
t1 = time.time()
deterministic([datetime(2016, 1, 1)], 12, model, data, ZarrBackend(str(raw)))
print(f"72 h rollout (1 init) in {time.time() - t1:.0f}s", flush=True)
ds = _EXTRACTION_FUNCS["interp"](raw, label, save_uv=False)
print(ds)
print(ds["wind_speed_10m"].isel(time=0).values[:, [0, 3, 11]])
shutil.rmtree(raw)
print(f"total {time.time() - t0:.0f}s  OK")
