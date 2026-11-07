# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit risk model comparison: Merton (1974) structural, Vasicek-Kealhofer (VK/KMV) structural, and Hull-White (2000) reduced-form intensity. The Hull-White hazard is bootstrapped directly from 5Y Markit CDS par spreads (not a logistic-regression proxy), with seniority-weighted firm-specific recovery rate (Altman & Kishore, 1996). Two evaluations on a 2021–2025 U.S. listed-firm panel: (i) structural discrimination on a 163-firm labelled panel (661 firm-years, 21 distressed firms / 67 distressed firm-years) scored by ROC-AUC / CAP-AR; (ii) CDS market calibration on the Markit IG extract (84 firms / 398 firm-years, of which 83 firms / 314 firm-years merge onto the structural panel) using λ_market and EL benchmarks, plus walk-forward linear calibration (Task A-2), cross-sectional Spearman vs CDS hazard (Task B), and Spearman vs agency ratings (Task C). The canonical deliverable is `docs/Grp03_mf728_report.pdf`.

## Commands

```bash
# Install dependencies
pip install -e .
# or
pip install numpy scipy pandas yfinance fredapi matplotlib scikit-learn openpyxl jupyter

# Run the full analysis (loads cached panel_data.csv + cds_parspread.csv)
jupyter notebook notebooks/main.ipynb

# Rebuild raw inputs (only when source data changes)
python src/scripts/filter_optionmetrics.py <raw_zip> data_raw/iv_YYYY.csv
python src/scripts/filter_optionmetrics.py --merge      # combine yearly IV files
python src/scripts/filter_cds.py                        # filter Markit dump → data_raw/cds_parspread.csv

# Rebuild the merged panel from yfinance + FRED + filtered IV
PYTHONPATH=src python -c "from data import build_panel; build_panel()"
```

## Architecture

```
src/                       — Python package (added to sys.path by the notebook)
  config.py                — Tickers, sample window, model parameters. FRED_API_KEY from env.
  data/                    — Live data acquisition modules (rebuild panel_data.csv)
    equity.py              — Stock prices, market cap, equity volatility (σ_E)
    balance_sheet.py       — Debt structure from yfinance financials
    risk_free.py           — Treasury rates from FRED (fallback: yfinance ^TNX)
    implied_vol.py         — Loads OptionMetrics implied vol from data_raw/
    merge.py               — Builds firm-level annual panel + default labels
  models/                  — Three credit risk model implementations
    merton.py              — Merton: scipy.optimize.fsolve for (V, σ_V) → DD → PD
    vk.py                  — VK/KMV: iterative asset value, DP = STD + 0.5×LTD
    hull_white.py          — Reduced-form: λ bootstrapped from 5Y CDS spreads (Hull & White, 2000); seniority-weighted RR (Altman & Kishore, 1996)
  analysis/                — Evaluation and visualization
    discrimination.py      — ROC/AUC, CAP/AR computation, model comparison
    visualization.py       — ROC curves, CAP curves, DD distributions, PD scatter
  scripts/                 — One-off raw-data preprocessing
    filter_cds.py          — Markit CDS dump → data_raw/cds_parspread.csv
    filter_optionmetrics.py — OptionMetrics opprcd → data_raw/iv_YYYY.csv → merge
data_raw/                  — Licensed WRDS inputs: gitignored, NOT redistributable
notebooks/main.ipynb       — End-to-end analysis pipeline
output/                    — Generated CSV files and PNG charts
  cds_calibration/         — Task A/B/C outputs (level matching, rank, rating benchmark)
docs/                      — Final report PDF and project proposal
report/                    — Slide sources and drafts (gitignored)
```

## Key Conventions

- All modules use `import config` for project-wide parameters; `src/` must be on `sys.path` (notebook handles this; for scripts use `PYTHONPATH=src` or `pip install -e .`)
- Each model's `run_*()` function takes a panel DataFrame and returns it with added columns (e.g., `merton_DD`, `merton_PD`, `hw_lambda`)
- Default label: `default_label` column (1 = distressed, 0 = non-default)
- Model output columns follow pattern: `{model}_{metric}` (e.g., `vk_DD`, `hw_PD_5Y`)
- Raw inputs live in `data_raw/` (gitignored: licensed Markit/OptionMetrics data, not redistributable); merged panel in `output/panel_data.csv`; analysis outputs in `output/` and `output/cds_calibration/`

## Setup Required

1. Set `FRED_API_KEY` as an environment variable (only needed when rebuilding the risk-free series). If unset, the system auto-falls back to yfinance `^TNX`.
2. Place filtered raw inputs in `data_raw/` (`cds_parspread.csv`, `implied_vol_optionmetrics.csv`). These are licensed WRDS extracts and are not in version control; regenerate them with `src/scripts/` (schema documented in README). The notebook needs only `cds_parspread.csv` at run time, plus the cached `output/panel_data.csv`.
