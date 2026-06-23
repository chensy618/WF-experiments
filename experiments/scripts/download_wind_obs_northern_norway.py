"""
Download hourly wind observations from the Frost API (frost.met.no) for nine
additional Northern Norway stations and save to CSV.

Stations
--------
SN90800  Torsvåg Fyr         (Karlsøy, Troms)       — coastal lighthouse
SN90720  Måsvik              (Tromsø, Troms)         — road/fjord station
SN91740  Sørkjosen Lufthavn  (Nordreisa, Troms)      — airport
SN89350  Bardufoss           (Målselv, Troms)        — airport (inland)
SN87110  Andøya              (Andøy, Nordland)       — airport
SN85380  Skrova Fyr          (Vågan, Nordland)       — coastal lighthouse
SN94500  Fruholmen Fyr       (Måsøy, Finnmark)       — coastal lighthouse
SN96400  Slettnes Fyr        (Gamvik, Finnmark)      — coastal lighthouse
SN95350  Banak               (Porsanger, Finnmark)   — airport

Years: 2016-2022 (one file per station per year)
Data availability verified: all stations have 744/744 hourly rows in January
for every year in this range.

Usage
-----
    python download_wind_obs_northern_norway.py

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
    "SN90800": "Torsvag_Fyr",
    "SN90720": "Masvik",
    "SN91740": "Sorkjosen_Lufthavn",
    "SN89350": "Bardufoss",
    "SN87110": "Andoya",
    "SN85380": "Skrova_Fyr",
    "SN94500": "Fruholmen_Fyr",
    "SN96400": "Slettnes_Fyr",
    "SN95350": "Banak",
}

YEAR_START = 2016
YEAR_END = 2022

ELEMENTS = "wind_speed,wind_from_direction,wind_speed_of_gust,max(wind_speed PT1H),min(wind_speed PT1H)"

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "wind_obs"

FROST_URL = "https://frost.met.no/observations/v0.jsonld"

CHUNK_MONTHS = 3
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

        for year in range(YEAR_START, YEAR_END + 1):
            out_path = OUTPUT_DIR / f"{station_id}_{station_name}_{year}.csv"
            if out_path.exists():
                print(f"  {year}: already exists, skipping.")
                continue

            print(f"  Fetching {year} …")
            year_frames: list[pd.DataFrame] = []

            for start, end in month_chunks(year, year, CHUNK_MONTHS):
                df = fetch_chunk(station_id, start, end)
                if not df.empty:
                    year_frames.append(df)
                time.sleep(REQUEST_DELAY)

            if not year_frames:
                print(f"  {year}: no data retrieved.")
                continue

            combined = pd.concat(year_frames, ignore_index=True)
            combined["time"] = pd.to_datetime(combined["time"], utc=True)
            combined.sort_values("time", inplace=True)
            combined.drop_duplicates(subset=["time", "element"], inplace=True)

            pivoted = combined.pivot_table(
                index="time", columns="element", values="value", aggfunc="first"
            ).reset_index()
            pivoted.columns.name = None

            pivoted.to_csv(out_path, index=False)
            print(f"  {year}: saved {len(pivoted)} hourly rows → {out_path}")


if __name__ == "__main__":
    main()
