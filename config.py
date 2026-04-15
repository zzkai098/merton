"""
Project configuration: sample firms, time window, API keys, model parameters.
"""

# ── FRED API Key ──────────────────────────────────────────────
# Get yours at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "672877af8d722519f7cbcd5f8c666532"

# ── Time Window ───────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# ── Sample Firms ──────────────────────────────────────────────
# Non-defaulted firms (~150, diverse sectors)
NON_DEFAULT_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "CRM", "ADBE",
    "AMD", "QCOM", "AVGO", "TXN", "NFLX", "PYPL", "INTC", "CSCO",
    "ORCL", "IBM", "NOW", "INTU", "AMAT", "MU", "LRCX", "KLAC",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BK", "AXP",
    "SCHW", "BLK", "SPGI", "USB", "PNC", "TFC", "MET", "AIG",
    "PRU", "ALL", "TRV", "CME",
    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT",
    "BMY", "GILD", "AMGN", "MDT", "CI", "CVS", "HUM", "BSX",
    "SYK", "ZTS", "REGN", "VRTX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
    "OXY", "HAL", "DVN", "WMB", "KMI",
    # Consumer Discretionary
    "HD", "LOW", "TGT", "COST", "NKE", "SBUX", "MCD", "YUM",
    "GM", "F", "TSLA", "LULU", "ROST", "TJX", "DHI", "LEN",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "CL", "KHC", "GIS", "SJM",
    "HSY", "MO", "PM", "STZ", "KR", "SYY",
    # Industrials
    "BA", "CAT", "GE", "MMM", "HON", "RTX", "LMT", "NOC",
    "DE", "EMR", "ITW", "WM", "RSG", "FDX", "UPS", "CSX",
    "UNP", "NSC",
    # Telecom & Media
    "T", "VZ", "CMCSA", "CHTR", "DIS", "WBD", "FOX",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE",
    # Airlines & Transport
    "DAL", "UAL", "LUV",
]

# Defaulted / severely distressed firms (~30)
# Format: (ticker, label, reason)
DEFAULT_FIRMS = [
    # Formal bankruptcy (Chapter 11)
    ("HTZ",   "Hertz",              "2020-05-22"),   # Ch.11 May 2020
    # Going-concern / severe operational distress
    ("PLUG",  "Plug Power",         "2023-12-31"),   # Going-concern warning
    ("FCEL",  "FuelCell Energy",    "2023-12-31"),   # Going-concern warning, persistent losses
    # removed NKLA — no balance sheet on yfinance
    # Severe financial distress — cash burn / pre-profit
    ("LCID",  "Lucid Group",        "2023-12-31"),   # Cash burn, high leverage
    ("RIVN",  "Rivian",             "2023-12-31"),   # Cash burn, pre-profit
    ("SPCE",  "Virgin Galactic",    "2023-12-31"),   # Cash burn, no revenue
    ("WKHS",  "Workhorse Group",    "2023-12-31"),   # Near-zero revenue, distressed
    # removed LAZR — no yfinance data
    ("QS",    "QuantumScape",       "2023-12-31"),   # Pre-revenue, cash burn
    ("BLNK",  "Blink Charging",    "2023-12-31"),   # Persistent losses
    # Severe financial distress — revenue decline / negative equity
    ("AMC",   "AMC Entertain.",     "2023-12-31"),   # Negative equity, high debt
    ("BYND",  "Beyond Meat",        "2023-12-31"),   # Revenue decline, distressed
    ("TLRY",  "Tilray Brands",      "2023-12-31"),   # Persistent losses
    ("CLOV",  "Clover Health",      "2023-12-31"),   # Persistent losses
    # removed WISH — no yfinance data
    ("PTON",  "Peloton",            "2023-12-31"),   # Revenue decline, cash burn
    ("W",     "Wayfair",            "2023-12-31"),   # Persistent losses, high debt
    ("OPEN",  "Opendoor Tech",      "2023-12-31"),   # Massive losses
    ("SKLZ",  "Skillz",             "2023-12-31"),   # Revenue collapse
    ("LMND",  "Lemonade",           "2023-12-31"),   # Persistent losses
    # removed RIDE, SWN — no yfinance data
    ("RIG",   "Transocean",         "2023-12-31"),   # High leverage, persistent losses
    ("CLF",   "Cleveland-Cliffs",   "2023-12-31"),   # Cyclical distress, high leverage
    ("AAL",   "American Airlines",  "2023-12-31"),   # Negative equity, high debt
]

DEFAULT_TICKERS = [t[0] for t in DEFAULT_FIRMS]
ALL_TICKERS = NON_DEFAULT_TICKERS + DEFAULT_TICKERS

# ── Model Parameters ──────────────────────────────────────────
RISK_FREE_SERIES = "DGS1"       # FRED 1-Year Treasury Constant Maturity
T = 1.0                         # Default horizon in years
TRADING_DAYS = 252              # Annualization factor
VOL_WINDOW = 252                # Rolling window for equity vol estimation
USE_IMPLIED_VOL = True          # Use option-implied vol for σ_E (fallback: historical)
IMPLIED_VOL_TARGET_TENOR = 1.0  # Target option tenor in years for implied vol
RECOVERY_RATE = 0.4             # LGD = 1 - recovery (standard assumption)

# ── Output ────────────────────────────────────────────────────
OUTPUT_DIR = "output"
