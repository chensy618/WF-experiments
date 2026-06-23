"""
Download WeatherBench2 HRES t=0 analysis zarr from Google Cloud Storage.

Target: gs://weatherbench2/datasets/hres/1440x721/2016-2022-6h-1440x721.zarr
Output: /cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts/hres/0p25/2016-2022-6h-1440x721.zarr

This dataset contains the HRES operational analysis (t=0) at 6-hourly intervals
(00/06/12/18 UTC) for 2016-2022, at 0.25 degree resolution.

Note: This is not exactly the same as the ECMWF official analysis (type=an).
It is derived from HRES forecast step=0 (type=fc, step=0), which differs mainly
for accumulated variables (e.g. total precipitation = 0 at step=0). For wind
and other instantaneous variables, the difference is negligible.
"""

import sys
from pathlib import Path

import gcsfs

GCS_PATH = "weatherbench2/datasets/hres/1440x721/2016-2022-6h-1440x721.zarr"
LOCAL_DIR = Path("/cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts/hres/0p25")
LOCAL_PATH = LOCAL_DIR / "2016-2022-6h-1440x721.zarr"


def main() -> None:
    if (LOCAL_PATH / ".zmetadata").exists():
        print(f"Already complete: {LOCAL_PATH}")
        sys.exit(0)
    if LOCAL_PATH.exists():
        print(f"Partial download found at {LOCAL_PATH}, resuming...")


    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source : gs://{GCS_PATH}")
    print(f"Target : {LOCAL_PATH}")
    print("Connecting to GCS and starting copy (this may take several hours)...")

    fs = gcsfs.GCSFileSystem(token="anon")
    fs.get(GCS_PATH, str(LOCAL_PATH), recursive=True)

    print(f"\nDone. Saved to {LOCAL_PATH}")


if __name__ == "__main__":
    main()
