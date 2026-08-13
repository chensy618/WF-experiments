"""
Case study entry point.

Steps
-----
1. Load ERA5 data source   (local_era5.LocalERA5 — reads /cluster/work/.../era5_6h)
2. Load model weights      (forecasting.load_model)
3. Run forecasts           (forecasting.run_forecasts)

Metrics and figures are produced in the notebook (fcn3_case_study_analysis.ipynb).

Usage
-----
    # FCN3, all case-study years
    python case_study_analysis.py --model fcn3

    # GraphCast, specific years, local weights
    python case_study_analysis.py --model graphcast --year 2016 2022 \\
        --package-path /cluster/work/projects/nn8106k/siyan/graphcast_weights

    # 2-day/3-day (+72h) case study — settings come from config.py (NSTEPS_LONG,
    # LONG_OUT_TAG, LONG_DAILY_ONLY), not from flags, so both models stay in sync
    python case_study_analysis.py --model fcn3 --preset long
    python case_study_analysis.py --model graphcast --preset long \\
        --package-path /cluster/work/projects/nn8106k/siyan/graphcast_weights

    # Bilinear-interpolation spatial matching (stationbench-style), instead of
    # nearest-neighbor. Writes to a separate {model}_interp output dir, so it
    # never overwrites the existing nearest-neighbor forecasts above.
    python case_study_analysis.py --model fcn3 --preset long --extraction interp
    python case_study_analysis.py --model graphcast --preset long --extraction interp \\
        --package-path /cluster/work/projects/nn8106k/siyan/graphcast_weights

    # Also save u10m/v10m wind components alongside the scalar wind_speed_10m
    # (which is still saved, unchanged). Writes to a separate {...}_uv output
    # dir, so it never overwrites any existing scalar-only forecast run.
    python case_study_analysis.py --model fcn3 --preset long --extraction interp --save-uv
    python case_study_analysis.py --model graphcast --preset long --extraction interp --save-uv \\
        --package-path /cluster/work/projects/nn8106k/siyan/graphcast_weights
"""

import argparse

from config import ALL_YEARS, LONG_DAILY_ONLY, LONG_OUT_TAG, NSTEPS, NSTEPS_LONG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case study wind forecast pipeline")
    parser.add_argument(
        "--model", choices=["fcn3", "graphcast"], required=True,
        help="Prognostic model to run."
    )
    parser.add_argument(
        "--year", type=int, nargs="+", default=ALL_YEARS,
        help=f"Year(s) to process. Default: all case-study years {ALL_YEARS}.",
    )
    parser.add_argument(
        "--preset", choices=["default", "long"], default="default",
        help="'long' pulls nsteps/out-tag/daily-only from config.py's "
             f"NSTEPS_LONG={NSTEPS_LONG}, LONG_OUT_TAG='{LONG_OUT_TAG}', "
             f"LONG_DAILY_ONLY={LONG_DAILY_ONLY} — the 2-day/3-day case-study "
             "run. Overrides --nsteps/--out-tag/--daily-only when set.",
    )
    parser.add_argument(
        "--nsteps", type=int, default=NSTEPS,
        help=f"Number of 6-hour forecast steps (default {NSTEPS}). Ignored if --preset long.",
    )
    parser.add_argument(
        "--out-tag", type=str, default="",
        help="Suffix for the forecast output dir, e.g. '_72h'. Ignored if --preset long.",
    )
    parser.add_argument(
        "--daily-only", action="store_true",
        help="Force daily 00 UTC init (overrides GraphCast's default 6-hourly "
             "cadence; no effect on FCN3). Ignored if --preset long.",
    )
    parser.add_argument(
        "--package-path", type=str, default=None,
        help="Path to pre-downloaded model weights (skips GCS download).",
    )
    parser.add_argument(
        "--era5-dir", type=str, default=None,
        help="Override ERA5 NetCDF directory (default: /cluster/work/.../era5_6h).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing weekly forecast zarrs.",
    )
    parser.add_argument(
        "--extraction", choices=["nearest", "interp"], default="nearest",
        help="Spatial matching method for pulling station values out of the raw "
             "global-grid forecast: 'nearest' (default, unchanged from all prior "
             "runs) or 'interp' (bilinear interpolation to each station's exact "
             "coordinates, matching stationbench's approach). 'interp' writes to "
             "a separate {model_label}{out_tag}_interp output directory.",
    )
    parser.add_argument(
        "--save-uv", action="store_true",
        help="Also extract and save u10m/v10m wind components alongside the "
             "scalar wind_speed_10m (which is still saved, unchanged). Writes "
             "to a separate {model_label}{out_tag}{extraction_suffix}_uv output "
             "directory, so it never overwrites any existing scalar-only run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = args.year

    if args.preset == "long":
        nsteps, out_tag, daily_only = NSTEPS_LONG, LONG_OUT_TAG, LONG_DAILY_ONLY
    else:
        nsteps, out_tag, daily_only = args.nsteps, args.out_tag, args.daily_only

    print("=" * 70)
    print(f"Case study  |  model={args.model}  |  years={years}  |  preset={args.preset}")
    print(f"  nsteps={nsteps}  out_tag='{out_tag}'  daily_only={daily_only}  "
          f"extraction={args.extraction}  save_uv={args.save_uv}")
    print("=" * 70)

    from forecasting import load_data_source, load_model, run_forecasts

    data               = load_data_source(args.era5_dir)
    model, model_label = load_model(args.model, args.package_path)

    for year in years:
        run_forecasts(model, model_label, data, year,
                      nsteps=nsteps, overwrite=args.overwrite,
                      daily_only=daily_only, out_tag=out_tag,
                      extraction=args.extraction, save_uv=args.save_uv)



if __name__ == "__main__":
    main()
