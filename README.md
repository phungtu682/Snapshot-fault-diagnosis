# Snapshot-based fault diagnosis in distribution networks

Code, data and results for the paper
*Understanding the limits of snapshot-based fault diagnosis in distribution networks*
(preprint: `paper/manuscript_arxiv.pdf`).

The repository reproduces every number, table and figure of the paper: fault simulation on the
IEEE 13-, 34- and 123-node feeders (OpenDSS), the four feature sets (AGG, PPA, SEQ, FPS), the
classifier benchmark, the measurement-error framework (TVE accuracy classes), the
discriminability-margin placement (DMP) against twelve placement methods, the robustness tests,
faulted-zone identification, the abstention (reject) option and the validation on the public
DER-Fault data set.

## Measurement-error model

A device of the *x* % TVE accuracy class has a 99th-percentile total vector error of *x* %, with the
error shared equally between magnitude and angle: sigma = x / 3.035
(1 % class: sigma_m = 0.33 %, sigma_theta = 0.19 deg). The common reading of 1 % TVE as
sigma_m = 1 %, sigma_theta = 0.5 deg behaves like a 3 % class device. See `experiments/k4lib.py`
(`tve_class`, `tve_quantiles`).

## Layout

```
experiments/            simulation, features, placement and the experiments e1-e17
  simlib.py             OpenDSS fault-snapshot generator
  simlib_offline.py     drop-in replacement that serves precomputed grids (no OpenDSS needed)
  features.py           measurement model and feature sets
  placement.py          DMP and placement baselines
  placement2.py         bound-based DMP variant and ranking index B(S)
  k4lib.py, e01.py, n2_law.py   library for the notebooks (noise classes, distinguishability baseline, R_f* law)
  e*.py                 one script per experiment, each writes results/<name>.json
  data/feeders/         IEEE 13/34/123-node and Iowa 240-bus OpenDSS models
  data/sim/             simulated snapshot data sets (.npz)
  data/derfault/        DER-Fault in .npz form plus its metadata files
  extra/                precomputed noise-free design fault scenarios and feeder graphs
  results/              JSON results of e1-e17
notebooks/              N1-N5, N6v2, S, R2 and R3 (Kaggle notebooks) with their result files
figures/                figure scripts, figure data and all figures (pdf/, png/)
paper/                  LaTeX source and PDF of the preprint
```

## Paper item -> source

| Paper item | Script / notebook | Result |
|---|---|---|
| Table 2 (reference noise levels) | `experiments/k4lib.py` (`tve_quantiles`) | -- |
| Table 3 (feature sets, Cover-Hart bound) | `e1_ladder.py` | `results/e1_ladder.json` |
| Figure 4 (accuracy by R_f and TVE class) | `e3_rf_noise.py` | `results/e3_rf_noise.json` |
| Table 4 (1 % TVE class vs micro-PMU) | `e11_upmu_noise.py` | `results/e11_upmu_noise.json` |
| Table 5 (required accuracy class) | `notebooks/N2_detectability_law.ipynb`, `notebooks/S_sd_supplement.ipynb` | `notebooks/results_N1_N6/N2_spec.csv`, `notebooks/results_S/S_spec_sd.csv` |
| Table 6 (13 classifiers) | `e13_classifiers.py` | `results/e13_classifiers.json` |
| Figure 5 (all 364 three-bus subsets) | `e4_placement.py` | `results/e4_placement.json` |
| Table 7 (placement incl. distinguishability method) | `notebooks/N1_e01_placement.ipynb` | `notebooks/results_N1_N6/N1_comparison.csv` |
| Figure 6, Wilcoxon tests | `e4b_dmp.py`, `e14_placement_baselines.py`, `e16_dmp2.py` | `results/e4b_dmp.json`, `e14_*.json`, `e16_*.json` |
| Table 8 (conditions of Khattak and Choudhry) | `notebooks/S_sd_supplement.ipynb` | `notebooks/results_S/S_repro_sd.csv` |
| Table 9 (selection time) | `e14_placement_baselines.py` | `results/e14_placement_baselines.json` |
| Table 10 (DER-Fault) | `e9_derfault.py` | `results/e9_derfault.json` |
| Table 11 (robustness) | `e17_robust_sd.py` | `results/e17_robust_sd.json` |
| Table 12 (faulted zone) | `notebooks/N5_zone_identification.ipynb` | `notebooks/results_N1_N6/N5_*.csv` |
| Table 13 (reject option) | `notebooks/N6v2_reject_margin.ipynb`, `notebooks/S_sd_supplement.ipynb` | `notebooks/results_N6v2/`, `notebooks/results_S/S_reject_folds.csv` |
| Figure 3 (worked DMP example) | `figures/compute_dmp34.py`, `figures/fig_dmp34_mechanism.py` | `figures/data/dmp34.npz` |
| Inference time | `e6_timing.py` | `results/e6_timing.json` |

`N3_current_vs_voltage.ipynb` and `N4a/N4b` are supplementary (substation current and N-1
placement); they are not reported in the paper's tables.

## Revision R3 (November 2026)

The R3 notebooks rerun or extend the experiments for the revised manuscript. Results are in
`notebooks/results_R3/`.

| Part | Notebook | Result | Content |
|---|---|---|---|
| A | `R3_teacher_comments.ipynb` | `R3_A_ablation.json` | removal of f1-f4 from AGG at the 1 % TVE class |
| B | `R3_teacher_comments.ipynb` | `R3_B_table3.json` | Table 3 (seven classifiers x four feature sets, 2 seeds x 5 folds, Cover-Hart estimate) |
| C | `R3_teacher_comments.ipynb` | `R3_C_dmp_reference.json` | DMP, three sensors, FPS, 1 % TVE and micro-PMU |
| D | `R3_teacher_comments.ipynb` | `R3_D_n1_protocols.json` | loss of one sensor, with and without retraining |
| E | `R3_teacher_comments.ipynb` | `R3_E_cover_hart_vs_n.json` | Cover-Hart estimate against sample size |
| F | `R3_teacher_comments.ipynb` | `R3_F_rf1000.json` | fault resistance up to 1000 ohm |
| G | `R3_teacher_comments.ipynb` | `R3_G_der_ieee.json` | current-limited inverters on the IEEE feeders (cross-validation within each data set) |
| H | `R3H_inverter_gridcode.ipynb` | `R3_H_inverter_gridcode.json` | grid-code inverter fault strategies (supplementary) |
| T | `R3T_transfer_noleak.ipynb` | `R3_T_transfer_noleak.json` | model trained without inverters, tested on new faults with inverters (different random seed, overlap check) |
| I | `R3I_ece.ipynb` | `R3_I_ece.json` | expected calibration error |

The transfer columns of `R3_G_der_ieee.json` and `R3_H_inverter_gridcode.json` reuse the training seed
and are superseded by part T.

## Installation

Python 3.11 was used.

```
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

## Reproducing the results

The simulated data sets are included, so every experiment runs directly:

```
cd experiments
python e1_ladder.py           # writes results/e1_ladder.json
...
cd ../figures
python make_figs_journal.py   # figures 4-6 into pdf/ and png/
python fig_dmp34_mechanism.py # figure 3
```

Without OpenDSS, use the precomputed grids: `cp experiments/simlib_offline.py experiments/simlib.py`.
The notebooks in `notebooks/` were run on Kaggle (CPU, 4 cores); each finds the data by itself.

## External data (not redistributed in full)

* **DER-Fault** (Mohamed Razick and Musilek, Zenodo, https://doi.org/10.5281/zenodo.22904220,
  CC BY 4.0). The converted file `data/derfault/derfault.npz` is included; the raw
  `DERfaults_measurements.csv` (1.3 GB) is not.
* **LBNL micro-PMU data** used by `e10_lbnl_noise.py` to measure the micro-PMU noise level; the raw
  files are not included, the result is in `results/e10_lbnl_noise.json`.

## Licence

Code: MIT (see `LICENSE`). Simulated data and results: CC BY 4.0. The IEEE test feeder models and
DER-Fault keep their own licences.
