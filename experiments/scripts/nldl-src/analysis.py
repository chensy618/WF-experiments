"""
Step 4 — Analysis: load pre-computed forecasts, compute metrics, save figures.
Orchestrates data_loading, metrics, and visualization modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import IN_TRAINING_YEARS, LEAD_H, OUT_ROOT
from data_loading import build_merged
from metrics import build_metrics_table
from visualization import plot_metrics_by_year, plot_station_map


def run_analysis(model_label: str, years: list[int]) -> None:
    """Full analysis pipeline for one model across multiple years.

    For each year:
      - loads pre-computed forecast zarrs, HRES, and station obs
      - inner-joins the three datasets on (station, valid_time)
      - computes RMSE, MSE, Bias, RSE per station and all-stations

    Then saves:
      - station map with 0.25° grid lines
      - multi-year RMSE / Bias bar chart
      - metrics_summary.csv
    """
    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Step 4] Analysis")
    print(f"  Model  : {model_label}")
    print(f"  Years  : {years}")

    # ── Station map (independent of model/year) ───────────────────────────────
    print("\n  Plotting station map ...")
    plot_station_map(fig_dir)

    # ── Per-year metrics ──────────────────────────────────────────────────────
    col = f"{model_label.lower()}_wind"
    all_rows: list[pd.DataFrame] = []

    for year in years:
        fc_dir = OUT_ROOT / "forecasts" / model_label / str(year)
        label  = "(in-training)" if year in IN_TRAINING_YEARS else "(out-of-training)"
        print(f"\n  Year {year} {label}")

        merged = build_merged(year, fc_dir, model_label)
        if merged.empty:
            print("    No overlapping data — skipping.")
            continue

        print(f"    Merged rows : {len(merged):,}")

        model_cols = [c for c in [col, "hres_wind"] if c in merged.columns]
        mdf = build_metrics_table(merged, model_cols)
        mdf["year"]   = year
        mdf["period"] = "in-training" if year in IN_TRAINING_YEARS else "out-of-training"
        all_rows.append(mdf)

        # Quick per-year summary to stdout
        summary = mdf[mdf["Station"] == "All Stations"][["Model", "N", "RMSE", "MSE", "Bias", "RSE"]]
        print(summary.round(4).to_string(index=False))

    if not all_rows:
        print("  No metrics computed — check forecast paths.")
        return

    all_metrics = pd.concat(all_rows, ignore_index=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out = OUT_ROOT / "metrics_summary.csv"
    all_metrics.round(4).to_csv(csv_out, index=False)
    print(f"\n  Metrics saved → {csv_out}")

    # ── Multi-year bar charts ─────────────────────────────────────────────────
    print("  Plotting metrics by year ...")
    plot_metrics_by_year(all_metrics, fig_dir)

    print(f"\n  Done. Figures → {fig_dir}")
