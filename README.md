<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# WF-experiments

**A Station-Based Evaluation of Machine Learning-based Weather Forecasting Models in Northern Norway**

Siyan Chen, Lars Uebbing, Eirik Mikal Samuelsen, Georgios Leontidis, Arnt-Børre Salberg, Sébastien Lefèvre, Robert Jenssen, Kristoffer Wickstrøm

[Paper (arXiv)](https://arxiv.org/abs/2609.10564) |
[Project page](https://chensy618.github.io/WF-experiments/) |
[Notebooks](experiments/notebooks) |
[Scripts](experiments/scripts) |
[How to run](experiments/scripts/nldl-src/instruction.md)

</div>

This repository evaluates two machine-learning global forecast models, NVIDIA
**FourCastNet3 (FCN3)** and Google DeepMind **GraphCast**, against ECMWF's physics-based
**HRES** forecast. All three are scored against hourly 10 m wind observations at
**12 MET Norway stations** in Troms, Nordland and Finnmark. The evaluation covers lead
times from **+6 h to +72 h** over **2016–2022**, with an extension back to 2010–2015.

The repository is a fork of [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio),
which is used to run the AI models. All experiment code lives under
[`experiments/`](experiments).

![RMSE by lead time, station and year](https://chensy618.github.io/WF-experiments/assets/combined_leadtime_station_annual_PAPER.png)

*(a) RMSE against lead time; the shading shows the spread across stations. (b) RMSE per
station, grouped by terrain. (c) RMSE per year.*

## Key findings

| | FCN3 | GraphCast | HRES |
| :-- | :--: | :--: | :--: |
| Overall RMSE vs station obs (m/s) | 2.96 | 2.94 | **2.89** |
| High-wind RMSE, obs ≥ 10.8 m/s (m/s) | **5.77** | 6.01 | 6.05 |
| High-wind bias (m/s) | **−4.97** | −5.43 | −5.50 |
| Mean ACC vs station obs | 0.550 | 0.565 | **0.565** |
| Mean ACC vs ERA5 | 0.836 | **0.861** | 0.824 |
| Inference time, 3-day forecast (s) | 95.93 | 151.43 | — |
| Peak GPU / CPU memory (GB) | 52.66 / 65.2 | 38.00 / 59.1 | — |

- **Overall skill:** HRES is slightly ahead, but the differences are small next to an
  RMSE of about 2.7–3.2 m/s.
- **High winds:** FCN3 has the lowest RMSE and the smallest bias of the three.
  Every model still underpredicts strong winds by about 5 m/s.
- **ACC depends on the reference:** both AI models were trained on ERA5. Their lead in
  ACC against ERA5 disappears when they are scored against independent station
  observations.
- **No drop in skill after training:** the change in RMSE and ACC between the
  training years and 2018–2022 is not significant for either model.

Significance is tested with a station-cluster bootstrap and a temporal block bootstrap
(blocks of 1–14 days). The
[project page](https://chensy618.github.io/WF-experiments/) has the full details.

## Repository layout

```text
experiments/
├── scripts/
│   ├── nldl-src/          # forecasting, ERA5 streaming, case-study driver, cost benchmark
│   ├── download_wind_obs_northern_norway.py
│   └── download_era5_6h.py, download_hres_forecast_wb2.py, ...
├── jobs/                  # SLURM scripts (FCN3 / GraphCast runs, data downloads)
├── notebooks/             # metrics, significance tests, figures
└── outputs/wind_obs/      # station observations (MET Norway)
earth2studio/              # upstream Earth2Studio package
```

## Reproduce

1. **Set up the environments.** FCN3 and GraphCast need separate environments. See
   [`requirements_fcn3_olivia.txt`](requirements_fcn3_olivia.txt) and
   [`requirements_graphcast_olivia.txt`](requirements_graphcast_olivia.txt).
2. **Download the data.** Get 6-hourly ERA5, the HRES forecasts from WeatherBench2 and
   the station observations with the scripts in
   [`experiments/scripts/`](experiments/scripts).
3. **Run the forecasts.** These are array jobs with one task per year:

   ```bash
   sbatch experiments/jobs/run_case_study_fcn3_72h_interp.slurm
   sbatch experiments/jobs/run_case_study_graphcast_72h_interp.slurm
   ```

4. **Analyse.** Compute the metrics and figures in
   [`model_comparison_analysis_72h_interpolate.ipynb`](experiments/notebooks/model_comparison_analysis_72h_interpolate.ipynb).
   The significance tests are in
   [`rebuttal_significance_tests.ipynb`](experiments/notebooks/rebuttal_significance_tests.ipynb).

Step-by-step instructions are in
[`experiments/scripts/nldl-src/instruction.md`](experiments/scripts/nldl-src/instruction.md).

## Citation

If you use this code or results, please cite:

```bibtex
@misc{chen2026station,
  title         = {A Station-Based Evaluation of Machine Learning-based Weather Forecasting Models in Northern Norway},
  author        = {Chen, Siyan and Uebbing, Lars and Samuelsen, Eirik Mikal and Leontidis, Georgios and Salberg, Arnt-B{\o}rre and Lef{\`e}vre, S{\'e}bastien and Jenssen, Robert and Wickstr{\o}m, Kristoffer},
  year          = {2026},
  eprint        = {2609.10564},
  archivePrefix = {arXiv},
  primaryClass  = {physics.ao-ph},
  url           = {https://arxiv.org/abs/2609.10564}
}
```

## Acknowledgements and license

This repository is built on [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio).
For Earth2Studio itself, see its
[documentation](https://nvidia.github.io/earth2studio/). The code is released under
the Apache-2.0 [license](LICENSE). Model weights and datasets keep their providers'
licences: FourCastNet3 (NVIDIA), GraphCast (Google DeepMind), HRES and ERA5 (ECMWF /
Copernicus), and station observations (MET Norway).
