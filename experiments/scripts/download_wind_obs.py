"""
Download hourly wind observations from the Frost API (frost.met.no) for three
Troms stations and save to CSV.

Stations
--------
SN88690  Hekkingen Fyr   (Senja, Troms)
SN90760  Fakken          (Karlsøy, Troms)
SN90490  Tromsø-langnes  (Tromsø, Troms)

Years: 2009-2012, 2022-2025

Usage
-----
    python download_wind_obs.py

Credentials are loaded from .env in the repo root (gitignored).
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLIENT_ID = os.environ.get("FROST_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("FROST_CLIENT_SECRET", "")

STATIONS = {
    "SN88690": "Hekkingen_Fyr",
    "SN90760": "Fakken",
    "SN90490": "Tromsoe_Langnes",
}

YEAR_RANGES = [(2009, 2012), (2022, 2025)]

ELEMENTS = "wind_speed,wind_from_direction,wind_speed_of_gust,max(wind_speed PT1H),min(wind_speed PT1H)"

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "wind_obs"

FROST_URL = "https://frost.met.no/observations/v0.jsonld"

# Frost API returns at most ~100 000 rows per request; chunk by month to be safe.
CHUNK_MONTHS = 3

# Seconds to wait between requests to stay within rate limits.
REQUEST_DELAY = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_chunk(station_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one time chunk for one station. Returns empty DataFrame on failure."""
    params = {
        "sources": station_id,
        "elements": ELEMENTS,
        "referencetime": f"{start}/{end}",
        "timeresolutions": "PT1H",
    }
    resp = requests.get(FROST_URL, params=params, auth=(CLIENT_ID, CLIENT_SECRET), timeout=60)

    if resp.status_code == 404:
        # No data for this period — not an error.
        return pd.DataFrame()

    if not resp.ok:
        print(f"    Warning: {resp.status_code} for {station_id} {start}–{end}: "
              f"{resp.json().get('error', {}).get('message', '')}")
        return pd.DataFrame()

    data = resp.json().get("data", [])
    if not data:
        return pd.DataFrame()

    rows = []
    for obs in data:
        ref_time = obs["referenceTime"]
        for o in obs.get("observations", []):
            rows.append({
                "time": ref_time,
                "element": o["elementId"],
                "value": o.get("value"),
                "unit": o.get("unit"),
                "quality_flag": o.get("qualityCode"),
            })

    return pd.DataFrame(rows)


def month_chunks(year_start: int, year_end: int, chunk_months: int):
    """Yield (start_str, end_str) pairs covering year_start–year_end inclusive."""
    periods = pd.period_range(
        start=f"{year_start}-01", end=f"{year_end}-12", freq="M"
    )
    for i in range(0, len(periods), chunk_months):
        chunk = periods[i : i + chunk_months]
        start = chunk[0].start_time.strftime("%Y-%m-%dT%H:%M:%S")
        # end is exclusive in Frost — use start of *next* period
        end_period = chunk[-1] + 1
        end = end_period.start_time.strftime("%Y-%m-%dT%H:%M:%S")
        yield start, end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not CLIENT_ID:
        raise SystemExit(
            "FROST_CLIENT_ID not set. Add it to .env in the repo root or export it."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for station_id, station_name in STATIONS.items():
        print(f"\n=== {station_name} ({station_id}) ===")
        all_frames: list[pd.DataFrame] = []

        for y_start, y_end in YEAR_RANGES:
            print(f"  Fetching {y_start}–{y_end} …")
            for start, end in month_chunks(y_start, y_end, CHUNK_MONTHS):
                df = fetch_chunk(station_id, start, end)
                if not df.empty:
                    all_frames.append(df)
                time.sleep(REQUEST_DELAY)

        if not all_frames:
            print(f"  No data retrieved for {station_id}.")
            continue

        combined = pd.concat(all_frames, ignore_index=True)
        combined["time"] = pd.to_datetime(combined["time"], utc=True)
        combined.sort_values("time", inplace=True)
        combined.drop_duplicates(subset=["time", "element"], inplace=True)

        # Pivot so each element is its own column
        pivoted = combined.pivot_table(
            index="time", columns="element", values="value", aggfunc="first"
        ).reset_index()
        pivoted.columns.name = None

        out_path = OUTPUT_DIR / f"{station_id}_{station_name}.csv"
        pivoted.to_csv(out_path, index=False)
        print(f"  Saved {len(pivoted)} hourly rows → {out_path}")


if __name__ == "__main__":
    main()
