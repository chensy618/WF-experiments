"""
ARCOChunkStream -- earth2studio-compatible ERA5 data source that streams ARCO-ERA5 from
GCS one *chunk* at a time. Drop-in replacement for LocalERA5 after the /cluster/work
purge (2026-11-10).

Why not earth2studio.data.ARCO: ARCO-ERA5 stores each pressure variable as one
blosc-lz4 chunk of all 37 levels per hour (~118 MB), and earth2studio.data.ARCO reads each
variable-level (z150, u700, ...) separately, so one FCN3 init downloads every 37-level
chunk 13 times (~7.7 GB per init). With ~10+ GPU tasks sharing the nodes' proxy this
saturated at ~400 s per init. Here every needed chunk is fetched once with concurrent raw
GETs (gcsfs cat) and decoded locally; all requested levels are sliced from it
(~0.6 GB per FCN3 init). Values are the same ERA5 fields bit for bit.

Output matches LocalERA5: DataArray (time, variable, lat, lon), float32, lat ascending.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import gcsfs
import numpy as np
import xarray as xr
from numcodecs import Blosc

from local_era5 import _prep_inputs

BUCKET = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

_SINGLE = {
    "u10m":  "10m_u_component_of_wind",
    "v10m":  "10m_v_component_of_wind",
    "u100m": "100m_u_component_of_wind",
    "v100m": "100m_v_component_of_wind",
    "t2m":   "2m_temperature",
    "msl":   "mean_sea_level_pressure",
    "sp":    "surface_pressure",
    "tcwv":  "total_column_water_vapour",
}
_PRESSURE = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
    "w": "vertical_velocity",
}
_PRESSURE_RE = re.compile(r"^([uvztqw])(\d+)$")


class ARCOChunkStream:
    """Stream ARCO-ERA5 chunks once per (variable, hour); slice levels locally."""

    def __init__(self) -> None:
        self._fs = gcsfs.GCSFileSystem(token="anon")
        ds = xr.open_zarr(f"gs://{BUCKET}", storage_options={"token": "anon"}, chunks=None)
        self._time = ds["time"].values
        self._level = [int(x) for x in ds["level"].values]
        lat = ds["latitude"].values
        self._lon = ds["longitude"].values
        self._flip = bool(lat[0] > lat[-1])
        self._lat = lat[::-1] if self._flip else lat
        self._meta: dict[str, dict] = {}

    def _zarray(self, name: str) -> dict:
        if name not in self._meta:
            m = json.loads(self._fs.cat(f"{BUCKET}/{name}/.zarray"))
            if m["compressor"]["id"] != "blosc" or m["filters"] or m["dtype"] != "<f4":
                raise ValueError(f"unexpected encoding for {name}: {m}")
            self._meta[name] = m
        return self._meta[name]

    def __call__(self, time, variable) -> xr.DataArray:
        times, variables = _prep_inputs(time, variable)
        return xr.concat([self._fetch_one(t, variables) for t in times], dim="time")

    def _fetch_one(self, t: datetime, variables: list[str]) -> xr.DataArray:
        ti = int(np.searchsorted(self._time, np.datetime64(t, "ns")))
        if ti >= len(self._time) or self._time[ti] != np.datetime64(t, "ns"):
            raise KeyError(f"{t} not in ARCO-ERA5 time axis")

        need: dict[str, list[tuple[str, int | None]]] = {}
        for v in variables:
            m = _PRESSURE_RE.match(v)
            if v in _SINGLE:
                need.setdefault(_SINGLE[v], []).append((v, None))
            elif m and int(m.group(2)) in self._level:
                need.setdefault(_PRESSURE[m.group(1)], []).append((v, int(m.group(2))))
            else:
                raise KeyError(f"No ARCO-ERA5 mapping for variable {v!r}")

        keys = {}
        for name, req in need.items():
            meta = self._zarray(name)
            nd = len(meta["shape"])
            keys[name] = f"{BUCKET}/{name}/{ti}" + ".0" * (nd - 1)

        raw = self._fs.cat(list(keys.values()))          # concurrent GETs, one per chunk
        got: dict[str, np.ndarray] = {}
        for name, req in need.items():
            meta = self._zarray(name)
            buf = raw[keys[name]]
            arr = np.frombuffer(Blosc().decode(buf), dtype="<f4").reshape(meta["chunks"])[0]
            for v, lv in req:
                a = arr[self._level.index(lv)] if lv is not None else arr
                got[v] = a[::-1] if self._flip else a

        data = np.stack([got[v] for v in variables]).astype(np.float32)[None]
        return xr.DataArray(data, dims=["time", "variable", "lat", "lon"],
                            coords={"time": [np.datetime64(t, "ns")], "variable": variables,
                                    "lat": self._lat, "lon": self._lon})
