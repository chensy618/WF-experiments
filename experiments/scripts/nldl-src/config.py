"""
Shared configuration: paths, station metadata, year classifications, colours.
All other modules import from here — nothing else imports from this file.
"""

from pathlib import Path

# =============================================================================
# Forecast settings
# =============================================================================

LEAD_H = 12    # hours — lead time used for evaluation
NSTEPS = 4     # 6-hour steps → +6 h, +12 h, +18 h, +24 h

# =============================================================================
# Year classification
# =============================================================================

IN_TRAINING_YEARS     = [2016, 2017]
OUT_OF_TRAINING_YEARS = [2020, 2021, 2022]
ALL_YEARS             = IN_TRAINING_YEARS + OUT_OF_TRAINING_YEARS

# =============================================================================
# Paths
# =============================================================================

_WORK = Path("/cluster/work/projects/nn8106k/siyan")
_REPO = Path("/cluster/home/siyan/github/WF-experiments")

HRES_PATH = _WORK / "weatherbench2_forecasts/hres/0p25/2016-2022-0012-1440x721.zarr"
OBS_DIR   = _REPO / "experiments/outputs/wind_obs"
OUT_ROOT  = _WORK / "WF-experiments/case-study"   # forecasts + figures land here

# =============================================================================
# Station metadata (12 stations, all in distinct 0.25° grid cells)
# =============================================================================

STATIONS: list[dict] = [
    # Troms
    {"id": "SN88690", "name": "Hekkingen Fyr",     "lat": 69.6005, "lon": 17.8317, "county": "Troms",    "obs_base": "SN88690_Hekkingen_Fyr"},
    {"id": "SN90490", "name": "Tromsø-Langnes",     "lat": 69.6767, "lon": 18.9133, "county": "Troms",    "obs_base": "SN90490_Tromsoe_Langnes"},
    {"id": "SN90760", "name": "Fakken",             "lat": 70.1043, "lon": 20.1145, "county": "Troms",    "obs_base": "SN90760_Fakken"},
    {"id": "SN90800", "name": "Torsvåg Fyr",        "lat": 70.2450, "lon": 19.5000, "county": "Troms",    "obs_base": "SN90800_Torsvag_Fyr"},
    {"id": "SN90720", "name": "Måsvik",             "lat": 69.9900, "lon": 18.6940, "county": "Troms",    "obs_base": "SN90720_Masvik"},
    {"id": "SN91740", "name": "Sørkjosen Lufthavn", "lat": 69.7900, "lon": 20.9520, "county": "Troms",    "obs_base": "SN91740_Sorkjosen_Lufthavn"},
    {"id": "SN89350", "name": "Bardufoss",          "lat": 69.0580, "lon": 18.5440, "county": "Troms",    "obs_base": "SN89350_Bardufoss"},
    # Nordland
    {"id": "SN87110", "name": "Andøya",             "lat": 69.3070, "lon": 16.1310, "county": "Nordland", "obs_base": "SN87110_Andoya"},
    {"id": "SN85380", "name": "Skrova Fyr",         "lat": 68.1530, "lon": 14.6490, "county": "Nordland", "obs_base": "SN85380_Skrova_Fyr"},
    # Finnmark
    {"id": "SN94500", "name": "Fruholmen Fyr",      "lat": 71.0940, "lon": 23.9840, "county": "Finnmark", "obs_base": "SN94500_Fruholmen_Fyr"},
    {"id": "SN96400", "name": "Slettnes Fyr",       "lat": 71.0890, "lon": 28.2170, "county": "Finnmark", "obs_base": "SN96400_Slettnes_Fyr"},
    {"id": "SN95350", "name": "Banak",              "lat": 70.0600, "lon": 24.9790, "county": "Finnmark", "obs_base": "SN95350_Banak"},
]

STATIONS_BY_ID: dict[str, dict] = {s["id"]: s for s in STATIONS}

# =============================================================================
# Colours
# =============================================================================

CLR = {
    "FCN3":      "#E74C3C",
    "GraphCast": "#2980B9",
    "HRES":      "#E67E22",
    "ERA5":      "#8E44AD",
    "Obs":       "#27AE60",
}

COUNTY_CLR = {
    "Troms":    "#2980B9",
    "Nordland": "#E74C3C",
    "Finnmark": "#27AE60",
}
