# Landslide susceptibility mapping in a plateau-mountain transition zone (IC-RF coupling model)

Reproducible analysis pipeline for the manuscript:

> **Coupling information content and random forest for landslide susceptibility mapping in a plateau-mountain transition zone: a case study of Yuxi City, Yunnan, China**
> (PLOS ONE, PONE-D-26-33050, revision)

This repository contains the code and derived data needed to reproduce every
number reported in the manuscript (Table 2, Table 3, and the evaluation
results in Section 3). All manuscript figures are regenerated locally by the
script and are not stored in the repository.

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
| `README.md` | This file. |

The derived data files (`results.txt`, `table2_values.csv`,
`ic_values_table.csv`) are stored so that reviewers can cross-check reported
numbers without running the code.

Files generated when you run the script (kept out of the repository to keep
it lightweight):
- `fig1.png` ... `fig8.png` (300 dpi; Fig 2 at 600 dpi)
- Intermediate single-panel outputs used to assemble the composite
  manuscript figures (`fig_roc_curves.png`, `fig_shap_importance.png`,
  `fig_shap_beeswarm.png`, `fig_cv_roc.png`)

## Raw data and original rasters

- The raw landslide inventory was compiled from local government records and
  field surveys; it is available from the corresponding author upon
  reasonable request, subject to data-sharing agreements.
- Original conditioning-factor rasters are publicly available from USGS
  (elevation derivatives), the National Tibetan Plateau Data Center
  (lithology, NDVI, precipitation), and OpenStreetMap (roads, rivers,
  faults). See the manuscript for details.

## Requirements

Python 3.7+, numpy, pandas, scikit-learn, matplotlib.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python rerun_model.py
```

The script writes all outputs into the current folder (see table above).

## Reproducibility notes

- The random forest uses a fixed `random_state=42`, so runs are
  deterministic given the same input CSV files.
- Figures use Times New Roman (PLOS style); if it is not installed on your
  machine the script falls back to Liberation Serif or DejaVu Serif
  (metrically compatible), which does not change any reported value.
- `best_rf_params.csv` is optional: if absent, the tuned hyperparameters
  hard-coded in `rerun_model.py` are used instead.

## License

MIT
