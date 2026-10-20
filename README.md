# Landslide susceptibility mapping in a plateau-mountain transition zone (IC-RF coupling model)

Reproducible analysis pipeline for the manuscript:

> **Coupling information content and random forest for interpretable landslide
> susceptibility mapping: a case study of Yuxi City, China**

This repository contains the code and derived data needed to reproduce every
number reported in the manuscript (Table 2, Table 3, and the evaluation
results in Section 3). All manuscript figures are regenerated locally by the
script and are not stored in the repository.

Repository: <https://github.com/wangtao11987/yuxi-landslide-susceptibility>

## Repository contents

| File | Description |
|---|---|
| `rerun_model.py` | Main analysis script (**required**). Re-runs the full IC, RF and IC-RF pipeline and regenerates all reported metrics, tables and figures. |
| `train_dataset_base.csv` | Training samples: nine conditioning factors + landslide label (0/1). No coordinates (**required**). |
| `test_dataset_base.csv` | Test samples: nine conditioning factors + landslide label (0/1). No coordinates (**required**). |
| `SHAP_Values_Base.csv` | SHAP values computed on the test set (**required**; data source for the SHAP beeswarm / importance panels). |
| `best_rf_params.csv` | Tuned random forest hyperparameters (**optional**; used if present, otherwise the defaults hard-coded in `rerun_model.py` are used). |
| `ic_values_table.csv` | Per-class information content values for all factors (data source for Fig 5 / Section 3.2). |
| `table2_values.csv` | TOL / VIF per conditioning factor, derived from the same correlation matrix used for the Fig 2 heatmap (data source for Table 2). |
| `results.txt` | Reference output of `rerun_model.py`; every reported number in the manuscript can be checked against this file. |
| `requirements.txt` | Python package dependencies. |
| `LICENSE` | MIT License. |
| `README.md` | This file. |

The derived data files (`results.txt`, `table2_values.csv`,
`ic_values_table.csv`) are stored so that reviewers can cross-check reported
numbers without running the code.

The anonymized sample datasets are the same 1620 sample points used in the
manuscript: 1134 training samples (567 positive / 567 negative) and 486 test
samples (243 positive / 243 negative), a 70 / 30 stratified split. Sample
coordinates are not included, so the files cannot be used to recover the
locations of individual landslides.

Files generated when you run the script (kept out of the repository to keep
it lightweight):
- `fig1.png` ... `fig8.png` (300 dpi; Fig 2 at 600 dpi)
- Intermediate single-panel outputs used to assemble the composite
  manuscript figures (`fig_roc_curves.png`, `fig_shap_importance.png`,
  `fig_shap_beeswarm.png`, `fig_cv_roc.png`)

## Data availability

This section mirrors the Data Availability statement in the manuscript.

**What is in this repository.** The analysis code, the anonymized sample
datasets, the SHAP values, the tuned hyperparameters, the per-class
information content values, the TOL / VIF table, and the derived results
supporting the reported model evaluations are publicly available here and at
<https://github.com/wangtao11987/yuxi-landslide-susceptibility>.

**What is not in this repository, and why.**

- *Landslide inventory.* The inventory was compiled from the historical
  geohazard inventory database maintained by local government agencies and
  from field surveys. It was provided under a data-sharing agreement that
  does not permit public redistribution, and publishing georeferenced
  landslide locations would disclose sensitive slope-stability information
  for populated areas of Yuxi City. The inventory is therefore available from
  the corresponding author on reasonable request, subject to the approval of
  the providing agency.
- *Original conditioning-factor rasters.* These are third-party products that
  are already openly available from their original providers (see below), so
  they are not re-deposited here.

## Conditioning factors and their original sources

The nine conditioning factors used in the models, and where the underlying
raster data come from. Full citations are given in the manuscript's Data
Sources section and Table 1.

| Factor(s) derived | Source data | Resolution / scale | Provider |
|---|---|---|---|
| Elevation, slope, aspect | SRTM GL1 DEM | 30 m | USGS |
| NDVI | Landsat 8 OLI Surface Reflectance | 30 m | USGS |
| Mean annual precipitation (MAP) | 1 km monthly precipitation dataset for China | 1 km | National Tibetan Plateau Data Center (TPDC) |
| Distance to faults | China Active Fault Database | 1:4,000,000 | National Earthquake Science Data Center |
| Distance to rivers, distance to roads | OpenStreetMap (OSM) vectors | Vector data | OpenStreetMap Foundation |
| Lithology | Regional Geological Map | 1:200,000 | Local government agencies |
| Landslide inventory | Historical geohazard inventory database | Point data | Local government agencies |

This table corresponds to Table 1 of the manuscript. The landslide inventory
and the lithology map are the two inputs supplied by local government
agencies and are therefore covered by the access conditions described above.

## Requirements

Python 3.7+, numpy, pandas, scikit-learn, matplotlib.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python rerun_model.py
```

The script writes all outputs into the current folder (see the table above).

## Reproducibility notes

- The random forest uses a fixed `random_state=42`, so runs are deterministic
  given the same input CSV files.
- The script uses Times New Roman for figure text if it is installed; if not,
  it falls back to Liberation Serif or DejaVu Serif (metrically compatible),
  which does not change any reported value.
- `best_rf_params.csv` is optional: if absent, the tuned hyperparameters
  hard-coded in `rerun_model.py` are used instead.
- `results.txt` records the expected output, including the TOL / VIF
  diagnostics (minimum TOL 0.817, maximum VIF 1.225), the IC model AUCs
  (0.7467 train / 0.7431 test), the RF model AUCs (0.8367 train / 0.7813
  test) and the IC-RF model AUCs (0.8058 train / 0.7703 test).

## License

This repository is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 Tao Wang.
