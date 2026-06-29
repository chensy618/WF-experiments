"""
Download WeatherBench2 HRES forecast zarr (0/12 UTC init) from Google Cloud Storage.
Only downloads 10m_wind_speed + coordinates.

Uses the public GCS HTTP API (no gcsfs required), which respects the standard
HTTPS_PROXY / https_proxy environment variables.

Source: gs://weatherbench2/datasets/hres_forecasts/1440x721/2016-2022-0012-1440x721.zarr
Output: /cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts/hres/0p25/2016-2022-0012-1440x721.zarr
"""

import shutil
import sys
import time
from pathlib import Path

import requests

BUCKET     = "weatherbench2"
ZARR_PATHS = [
    "datasets/hres/2016-2022-0012-1440x721.zarr",
    "datasets/hres_forecasts/1440x721/2016-2022-0012-1440x721.zarr",
]
NEEDED_VARS = [
    "10m_wind_speed",
    "time",
    "prediction_timedelta",
    "latitude",
    "longitude",
]

LOCAL_DIR  = Path("/cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts/hres/0p25")
LOCAL_PATH = LOCAL_DIR / "2016-2022-0012-1440x721.zarr"

GCS_STORAGE = f"https://storage.googleapis.com/{BUCKET}"
GCS_API     = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"

SESSION = requests.Session()
SESSION.headers["Accept-Encoding"] = "identity"


def _get(url: str, **kwargs) -> requests.Response:
    for attempt in range(5):
        try:
            resp = SESSION.get(url, timeout=120, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.HTTPError:
            raise  # 4xx/5xx — don't retry
        except requests.RequestException as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"  Retry {attempt+1}/5 after {wait}s: {e}", flush=True)
            time.sleep(wait)


def list_prefix(prefix: str) -> list:
    """List all object names under a GCS prefix (handles pagination)."""
    names = []
    params: dict = {"prefix": prefix, "maxResults": "1000"}
    while True:
        data = _get(GCS_API, params=params).json()
        names.extend(item["name"] for item in data.get("items", []))
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return names


def object_exists(obj_path: str) -> bool:
    try:
        _get(f"{GCS_API}/{requests.utils.quote(obj_path, safe='')}")
        return True
    except requests.HTTPError:
        return False


def find_zarr_path() -> str:
    # Try both .zgroup and .zmetadata — zarr v2 stores may have either
    for path in ZARR_PATHS:
        for marker in [".zmetadata", ".zgroup", ".zattrs"]:
            root_obj = f"{path}/{marker}"
            print(f"Checking gs://{BUCKET}/{root_obj} ...", flush=True)
            if object_exists(root_obj):
                print(f"Found: gs://{BUCKET}/{path}  (marker: {marker})")
                return path
    print("ERROR: zarr not found at any candidate path.")
    print("Listing available objects:")
    for base in ["datasets/hres_forecasts/1440x721/", "datasets/hres/1440x721/"]:
        try:
            items = list_prefix(base)[:10]
            for i in items:
                print(f"  gs://{BUCKET}/{i}")
        except Exception:
            pass
    sys.exit(1)


def download_object(gcs_obj: str, local_file: Path) -> None:
    url = f"{GCS_STORAGE}/{gcs_obj}"
    resp = _get(url, stream=True)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    with open(local_file, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
            f.write(chunk)


def main() -> None:
    wind_dir = LOCAL_PATH / "10m_wind_speed"
    if wind_dir.exists() and any(wind_dir.iterdir()):
        print(f"Already complete: {LOCAL_PATH}")
        sys.exit(0)

    if LOCAL_PATH.exists():
        print(f"Incomplete download found, removing: {LOCAL_PATH}")
        shutil.rmtree(LOCAL_PATH)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)

    # Show proxy being used (for debugging)
    import os
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "(none)"
    print(f"HTTPS proxy: {proxy}")

    zarr_path = find_zarr_path()
    print(f"\nSource : gs://{BUCKET}/{zarr_path}")
    print(f"Target : {LOCAL_PATH}")
    print(f"Vars   : {NEEDED_VARS}\n")

    # Root metadata files
    root_objects = list_prefix(f"{zarr_path}/.")
    # Also grab top-level metadata directly (some zarr versions store at root)
    for meta in [".zgroup", ".zattrs", ".zmetadata"]:
        obj = f"{zarr_path}/{meta}"
        local_f = LOCAL_PATH / meta
        if not local_f.exists() and object_exists(obj):
            print(f"  Metadata: {meta}")
            download_object(obj, local_f)

    # Download each needed variable/coordinate
    for var in NEEDED_VARS:
        prefix = f"{zarr_path}/{var}/"
        objects = list_prefix(prefix)
        if not objects:
            print(f"  WARNING: {var} — no objects found, skipping")
            continue
        print(f"  {var}: {len(objects)} chunks", flush=True)
        (LOCAL_PATH / var).mkdir(exist_ok=True)
        for i, obj in enumerate(objects):
            name       = obj[len(f"{zarr_path}/{var}/"):]
            local_file = LOCAL_PATH / var / name
            if local_file.exists():
                continue
            download_object(obj, local_file)
            if (i + 1) % 200 == 0:
                print(f"    {var}: {i+1}/{len(objects)} done", flush=True)
        print(f"  Done: {var}")

    print(f"\nDownload complete: {LOCAL_PATH}")


if __name__ == "__main__":
    main()
