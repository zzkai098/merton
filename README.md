# Comparing Structural and Reduced-Form Credit Risk Models

**Merton (1974) · VK/KMV (Crosbie & Bohn, 2003) · Hull-White (2000)**

Empirical comparison of three credit risk models on U.S. listed firms, evaluated by default discrimination ability using ROC/AUC and CAP/AR metrics.

---

## Results

| Model | AUC | AR |
|-------|-----|----|
| Merton (1974) | 0.910 | 0.819 |
| VK/KMV (Crosbie & Bohn, 2003) | 0.924 | 0.847 |
| **Hull-White (2000)** | **0.991** | **0.982** |

---

## Sample

- **50 U.S. listed firms** — 40 non-defaulted, 10 defaulted
- **Period**: 2018–2023
- **Panel**: 104 firm-year observations

Default firms: HTZ (Ch.11), PLUG (going-concern), TLRY, LCID, RIVN, BYND, SPCE, WKHS, CLOV, AMC (severe financial distress)

---

## Data Sources

| Source | Access | Variables |
|--------|--------|-----------|
| Yahoo Finance | `yfinance` | Daily prices, market cap, balance sheet |
| FRED | `fredapi` (fallback: `^TNX`) | 1Y Treasury rate (`DGS1`) |
| Manual labels | `config.py` | Default label (0/1) |

---

## Project Structure

```
config.py               — Tickers, time window, FRED API key, model parameters
data/
  equity.py             — Stock prices, market cap, equity volatility (σ_E)
  balance_sheet.py      — Debt structure from yfinance financials
  risk_free.py          — Treasury rates from FRED (fallback: yfinance ^TNX)
  merge.py              — Builds firm-level annual panel + default labels
models/
  merton.py             — Merton: scipy.fsolve for (V, σ_V) → DD → PD
  vk.py                 — VK/KMV: iterative asset estimation, DP = STD + 0.5×LTD
  hull_white.py         — Reduced-form: logistic regression hazard proxy
analysis/
  discrimination.py     — ROC/AUC, CAP/AR computation and model comparison
  visualization.py      — ROC curves, CAP curves, DD distributions, PD scatter
notebooks/main.ipynb    — End-to-end pipeline
output/                 — Generated CSV files and PNG charts
```

---

## Setup

```bash
pip install -e .
# or
pip install numpy scipy pandas yfinance fredapi matplotlib scikit-learn openpyxl jupyter
```

Set your FRED API key in `config.py`:

```python
FRED_API_KEY = "your_key_here"  # https://fred.stlouisfed.org/docs/api/api_key.html
```

## Run

```bash
jupyter notebook notebooks/main.ipynb
```

---

## Models

### Merton (1974) — Structural

Equity as a European call on firm assets. Solves two simultaneous equations for unobservable $(V, \sigma_V)$:

$$E = VN(d_1) - De^{-rT}N(d_2), \qquad \sigma_E E = VN(d_1)\sigma_V$$

$$DD = \frac{\ln(V/D) + (r - \sigma_V^2/2)T}{\sigma_V\sqrt{T}}, \qquad PD = N(-DD)$$

### VK/KMV (Crosbie & Bohn, 2003) — Structural

Two improvements over Merton: (1) default point $DP = STD + 0.5 \times LTD$; (2) iterative asset value estimation instead of simultaneous solve; (3) physical drift $\mu = r + 0.02$ in DD calculation.

### Hull-White (2000) — Reduced-Form

Models default intensity $\lambda$ directly, bypassing the unobservable $V$ problem:

$$PD = 1 - e^{-\lambda T}$$

$\lambda$ estimated via logistic regression on leverage ratio, equity volatility, and log market cap, with 5-fold cross-validation.

---

## References

- Merton, R. C. (1974). *Journal of Finance*, 29(2), 449–470.
- Black, F., & Scholes, M. (1973). *Journal of Political Economy*, 81(3), 637–654.
- Crosbie, P., & Bohn, J. (2003). *Modeling Default Risk*. Moody's KMV.
- Vassalou, M., & Xing, Y. (2004). *Journal of Finance*, 59(2), 831–868.
- Hull, J., & White, A. (2000). *Journal of Derivatives*, 8(1), 29–40.
- Duffie, D., & Singleton, K. J. (1999). *Review of Financial Studies*, 12(4), 687–720.
- Shumway, T. (2001). *Journal of Business*, 74(1), 101–124.
- Arora, N., Bohn, J. R., & Zhu, F. (2005). *Reduced form vs. structural models*. Moody's KMV.
