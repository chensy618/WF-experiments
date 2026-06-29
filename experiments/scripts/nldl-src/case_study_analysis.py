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
"""

import argparse

from config import ALL_YEARS, NSTEPS


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
        "--nsteps", type=int, default=NSTEPS,
        help=f"Number of 6-hour forecast steps (default {NSTEPS}).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = args.year

    print("=" * 70)
    print(f"Case study  |  model={args.model}  |  years={years}")
    print("=" * 70)

    from forecasting import load_data_source, load_model, run_forecasts

    data               = load_data_source(args.era5_dir)
    model, model_label = load_model(args.model, args.package_path)

    for year in years:
        run_forecasts(model, model_label, data, year,
                      nsteps=args.nsteps, overwrite=args.overwrite)



if __name__ == "__main__":
    main()
