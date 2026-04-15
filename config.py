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
    ("PLUG",  "Plug Power",         "2023-11-09"),   # 10-Q going-concern warning
    ("FCEL",  "FuelCell Energy",    "2023-12-19"),   # FY2023 10-K going-concern
    # Severe financial distress — cash burn / pre-profit
    ("LCID",  "Lucid Group",        "2025-02-25"),   # FY2024 10-K severe cash burn
    ("RIVN",  "Rivian",             "2022-05-11"),   # Q1 2022 quarterly loss $1.6B
    ("SPCE",  "Virgin Galactic",    "2025-02-26"),   # FY2024 10-K first going-concern
    ("WKHS",  "Workhorse Group",    "2024-03-12"),   # FY2023 10-K first going-concern
    ("QS",    "QuantumScape",       "2025-02-26"),   # Cash runway ~11 months
    ("BLNK",  "Blink Charging",    "2025-04-09"),   # FY2024 10-K going-concern
    # Severe financial distress — revenue decline / negative equity
    ("AMC",   "AMC Entertain.",     "2020-06-30"),   # Q2 2020 equity turned negative
    ("BYND",  "Beyond Meat",        "2022-06-30"),   # Q2 2022 first YoY revenue decline
    ("TLRY",  "Tilray Brands",      "2021-05-03"),   # Post-merger persistent losses
    ("CLOV",  "Clover Health",      "2021-01-07"),   # SPAC close, never profitable
    ("PTON",  "Peloton",            "2022-01-20"),   # Revenue decline, CEO replaced
    ("W",     "Wayfair",            "2019-09-30"),   # Q3 2019 equity turned negative
    ("OPEN",  "Opendoor Tech",      "2022-09-30"),   # Q3 2022 massive loss $928M
    ("SKLZ",  "Skillz",             "2022-03-31"),   # Q1 2022 revenue collapse began
    ("LMND",  "Lemonade",           "2020-07-02"),   # IPO with cumulative loss $250M
    # High leverage / cyclical distress
    ("RIG",   "Transocean",         "2015-12-31"),   # 2015 oil crash, net loss $3.1B
    ("CLF",   "Cleveland-Cliffs",   "2020-06-30"),   # Q2 2020 pandemic losses
    ("AAL",   "American Airlines",  "2020-03-31"),   # Q1 2020 equity turned negative
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
