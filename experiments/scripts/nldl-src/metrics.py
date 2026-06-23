"""
Metric computation: RMSE, MSE, Bias, RSE.
Pure functions — no I/O, no plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import STATIONS_BY_ID


def compute_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """Compute four scalar metrics for one (pred, obs) pair.

    Metrics
    -------
    RMSE  Root Mean Square Error  (m/s)
    MSE   Mean Square Error       (m²/s²)
    Bias  Mean Error              (m/s, positive = overestimate)
    RSE   Relative Squared Error  = MSE / Var(obs)  (dimensionless)
    """
    err    = pred - obs
    mse    = float(np.mean(err ** 2))
    rmse   = float(np.sqrt(mse))
    bias   = float(np.mean(err))
    var_obs = float(np.var(obs))
    rse    = mse / var_obs if var_obs > 0 else np.nan
    return {"RMSE": rmse, "MSE": mse, "Bias": bias, "RSE": rse}


def build_metrics_table(merged: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Compute metrics per station and all-stations for every model column.

    Parameters
    ----------
    merged      : DataFrame with columns station, obs_wind, and one or more model wind columns
    model_cols  : list of column names to evaluate, e.g. ["fcn3_wind", "hres_wind"]

    Returns
    -------
    DataFrame with columns: Station, Model, N, RMSE, MSE, Bias, RSE
    """
    rows: list[dict] = []

    station_groups = [(sid, merged[merged["station"] == sid]) for sid in STATIONS_BY_ID]
    station_groups.append(("All Stations", merged))

    for label, sub in station_groups:
        for col in model_cols:
            s = sub.dropna(subset=[col, "obs_wind"])
            if len(s) < 2:
                continue
            m = compute_metrics(s[col].values, s["obs_wind"].values)
            model_name = col.replace("_wind", "").upper()
            rows.append({"Station": label, "Model": model_name, "N": len(s), **m})

    return pd.DataFrame(rows)
