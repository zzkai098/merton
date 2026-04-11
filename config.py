"""
Project configuration: sample firms, time window, API keys, model parameters.
"""

# ── FRED API Key ──────────────────────────────────────────────
# Get yours at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "672877af8d722519f7cbcd5f8c666532"

# ── Time Window ───────────────────────────────────────────────
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"

# ── Sample Firms ──────────────────────────────────────────────
# Non-defaulted firms (diverse sectors, liquid large-caps)
NON_DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "JNJ",
    "JPM", "BAC", "WFC", "GS", "MS",
    "XOM", "CVX", "PFE", "UNH", "WMT",
    "PG", "KO", "PEP", "MCD", "DIS",
    "BA", "CAT", "GE", "MMM", "IBM",
    "T", "VZ", "INTC", "CSCO", "ORCL",
    "HD", "LOW", "TGT", "COST", "NKE",
    "FDX", "UPS", "DAL", "UAL", "LUV",
]

# Defaulted / severely distressed firms (confirmed yfinance data available)
# Format: (ticker, label, reason)
DEFAULT_FIRMS = [
    ("HTZ",   "Hertz",           "2020-05-22"),   # Ch.11 May 2020
    ("TLRY",  "Tilray Brands",   "2023-12-31"),   # Persistent losses, distressed
    ("LCID",  "Lucid Group",     "2023-12-31"),   # Cash burn, high leverage
    ("RIVN",  "Rivian",          "2023-12-31"),   # Cash burn, pre-profit
    ("BYND",  "Beyond Meat",     "2023-12-31"),   # Revenue decline, distressed
    ("SPCE",  "Virgin Galactic", "2023-12-31"),   # Cash burn, no revenue
    ("PLUG",  "Plug Power",      "2023-12-31"),   # Going-concern warning
    ("WKHS",  "Workhorse Group", "2023-12-31"),   # Near-zero revenue, distressed
    ("CLOV",  "Clover Health",   "2023-12-31"),   # Persistent losses
    ("AMC",   "AMC Entertain.",  "2023-12-31"),   # Negative equity, high debt
]

DEFAULT_TICKERS = [t[0] for t in DEFAULT_FIRMS]
ALL_TICKERS = NON_DEFAULT_TICKERS + DEFAULT_TICKERS

# ── Model Parameters ──────────────────────────────────────────
RISK_FREE_SERIES = "DGS1"       # FRED 1-Year Treasury Constant Maturity
T = 1.0                         # Default horizon in years
TRADING_DAYS = 252              # Annualization factor
VOL_WINDOW = 252                # Rolling window for equity vol estimation
RECOVERY_RATE = 0.4             # LGD = 1 - recovery (standard assumption)

# ── Output ────────────────────────────────────────────────────
OUTPUT_DIR = "output"
