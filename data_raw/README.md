# `data_raw/` — licensed inputs (not in version control)

The two raw inputs for this project are licensed WRDS products and are **not
redistributable**, so they are excluded from this repository via `.gitignore`.
Everything in this directory except this file is ignored.

| File | Source | Needed to |
|------|--------|-----------|
| `cds_parspread.csv` | WRDS → Markit CDS | Re-run `notebooks/main.ipynb` (Hull-White, §4 and §6) |
| `implied_vol_optionmetrics.csv` | WRDS → OptionMetrics Ivy DB US | Rebuild `output/panel_data.csv` from scratch |

The cached merged panel `output/panel_data.csv` is committed, so the structural
models (Merton, VK) in sections 1–5 of the notebook run without either file.
Only the Hull-White and CDS-calibration sections (§4, §6) require
`cds_parspread.csv`.

See the **Licensed data** section of the top-level `README.md` for the column
schema of both files and the exact commands to regenerate them with a WRDS
subscription.
