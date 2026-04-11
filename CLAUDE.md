# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit risk model comparison: Merton (1974) structural model, Vasicek-Kealhofer (VK/KMV) structural model, and Hull-White reduced-form intensity model. Evaluates default discrimination ability using ROC/AUC and CAP/AR metrics on U.S. listed firms.

## Commands

```bash
# Install dependencies
pip install -e .
# or
pip install numpy scipy pandas yfinance fredapi matplotlib scikit-learn openpyxl jupyter

# Run the full pipeline (from project root)
jupyter notebook notebooks/main.ipynb

# Run individual modules from project root
python -c "from data import build_panel; build_panel()"
python -c "from models import run_merton; import pandas as pd; run_merton(pd.read_csv('output/panel_data.csv'))"
```

## Architecture

```
config.py          — All configuration: ticker lists, time window, FRED API key, model parameters
data/              — Data acquisition from yfinance + FRED
  equity.py        — Stock prices, market cap, equity volatility (σ_E)
  balance_sheet.py — Debt structure from yfinance financials
  risk_free.py     — Treasury rates from FRED (fallback: yfinance ^TNX)
  merge.py         — Builds firm-level annual panel, adds default labels
models/            — Three credit risk model implementations
  merton.py        — Merton: scipy.optimize.fsolve for (V, σ_V), then DD → PD
  vk.py            — VK/KMV: iterative asset value estimation, default_point = STD + 0.5×LTD
  hull_white.py    — Reduced-form: logistic regression hazard proxy (no CDS data)
analysis/          — Evaluation and visualization
  discrimination.py — ROC/AUC, CAP/AR computation, model comparison report
  visualization.py  — ROC curves, CAP curves, DD distributions, PD scatter plots
notebooks/main.ipynb — End-to-end pipeline notebook
output/            — Generated CSV files and PNG charts
```

## Key Conventions

- All modules use `import config` for project-wide parameters (run from project root)
- Each model's `run_*()` function takes a panel DataFrame and returns it with added columns (e.g., `merton_DD`, `merton_PD`)
- Default label: `default_label` column (1 = defaulted, 0 = non-defaulted)
- Model output columns follow pattern: `{model}_{metric}` (e.g., `vk_DD`, `hw_PD`)

## Setup Required

1. Set your FRED API key in `config.py` (`FRED_API_KEY = "your_key_here"`)
2. If FRED key is unavailable, the system auto-falls back to yfinance ^TNX for risk-free rates
