"""
Visualisation: station map and multi-year metrics bar charts.
All functions save to fig_dir and close the figure — no plt.show() in scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import CLR, COUNTY_CLR, IN_TRAINING_YEARS, LEAD_H, STATIONS


def plot_station_map(fig_dir: Path) -> None:
    """Plot all 12 stations on a map with 0.25° model grid lines overlaid."""
    lon_min, lon_max = 13.0, 32.0
    lat_min, lat_max = 67.0, 72.5

    fig, ax = plt.subplots(figsize=(12, 7))

    # ── 0.25° grid lines ────────────────────────────────────────────────────
    res = 0.25
    for lon in np.arange(np.ceil(lon_min / res) * res, lon_max + res, res):
        ax.axvline(lon, color="lightgray", lw=0.5, zorder=1)
    for lat in np.arange(np.ceil(lat_min / res) * res, lat_max + res, res):
        ax.axhline(lat, color="lightgray", lw=0.5, zorder=1)

    # ── Stations ─────────────────────────────────────────────────────────────
    seen: set[str] = set()
    for st in STATIONS:
        clr   = COUNTY_CLR[st["county"]]
        label = st["county"] if st["county"] not in seen else None
        seen.add(st["county"])
        ax.scatter(st["lon"], st["lat"], color=clr, s=80, zorder=4,
                   edgecolors="black", linewidths=0.6, label=label)
        ax.annotate(
            f"{st['id']}\n{st['name']}",
            xy=(st["lon"], st["lat"]), xytext=(4, 4),
            textcoords="offset points", fontsize=6.5, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.7, edgecolor="none"),
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title("Northern Norway Evaluation Stations — 0.25° Model Grid",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="lightgray", lw=1.2))
    labels.append("0.25° model grid")
    ax.legend(handles, labels, title="Legend", fontsize=9,
              title_fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    plt.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "station_map_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_metrics_by_year(all_metrics: pd.DataFrame, fig_dir: Path) -> None:
    """Bar chart of RMSE and Bias per model per year.

    Parameters
    ----------
    all_metrics : DataFrame with columns Station, Model, year, RMSE, Bias
                  (output of metrics.build_metrics_table with year column added)
    fig_dir     : output directory
    """
    df    = all_metrics[all_metrics["Station"] == "All Stations"].copy()
    years  = sorted(df["year"].unique())
    models = df["Model"].unique()
    x, w   = np.arange(len(years)), 0.22
    n      = len(models)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    fig, (ax_r, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    for model, offset in zip(models, offsets):
        sub  = df[df["Model"] == model]
        rmse = [float(sub[sub["year"] == y]["RMSE"].iloc[0]) if y in sub["year"].values else np.nan for y in years]
        bias = [float(sub[sub["year"] == y]["Bias"].iloc[0]) if y in sub["year"].values else np.nan for y in years]
        clr  = CLR.get(model, "#888888")
        ax_r.bar(x + offset, rmse, w, label=model, color=clr, alpha=0.85)
        ax_b.bar(x + offset, bias, w, label=model, color=clr, alpha=0.85)

    for ax, ylabel, title in [
        (ax_r, "RMSE (m/s)", "RMSE vs Observations"),
        (ax_b, "Bias (m/s)", "Bias vs Observations"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for i, yr in enumerate(years):
            shade = "orange" if yr in IN_TRAINING_YEARS else "#AED6F1"
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.07, color=shade, zorder=0)

    ax_b.axhline(0, color="k", lw=0.8, linestyle="--", alpha=0.5)
    fig.suptitle(
        f"Forecast Error vs Station Observations  |  lead = +{LEAD_H} h\n"
        "Orange = in-training (2016, 2017)  |  Blue = out-of-training (2020–2022)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    out = fig_dir / "metrics_by_year_obs.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
