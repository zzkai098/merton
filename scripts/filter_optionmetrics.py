"""
OptionMetrics Option Data Filter
=================================
Each team member runs this script on their assigned year's zip file
to extract ATM call implied volatility for our 163 target tickers.

Usage:
    python scripts/filter_optionmetrics.py  <zip_path>  <output_csv_path>

Examples:
    python scripts/filter_optionmetrics.py  ~/Downloads/opprcd_2021.zip  data/iv_2021.csv
    python scripts/filter_optionmetrics.py  ~/Downloads/opprcd_2022.zip  data/iv_2022.csv

After all 5 members finish, combine the outputs:
    python scripts/filter_optionmetrics.py --merge

Filtering logic:
    1. Keep only our 163 target tickers
    2. Keep only the last 5 trading days of December each year (year-end)
    3. Keep only Call options (cp_flag = 'C')
    4. Keep only options with 250-450 days to expiry (approx. 1 year)
    5. Keep only ATM options (delta between 0.40 and 0.60)
    6. Remove invalid implied vol (NaN, zero, or > 3.0)
    7. Remove options with zero volume AND zero open interest (no trading)
    8. Per ticker-year, pick the row with delta closest to 0.50;
       break ties by days closest to 365
    9. Output one row per ticker-year

Note:
    - strike_price in OptionMetrics is actual strike * 1000; this script
      converts it back to the real strike (strike_price / 1000).
    - Requires: pandas, numpy (pip install pandas numpy)
"""

import sys
import os
import zipfile
import io
import pandas as pd
import numpy as np

# ── 163 target tickers ────────────────────────────────────────
TARGET_TICKERS = {
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
    # Defaulted / distressed
    "HTZ", "PLUG", "FCEL", "LCID", "RIVN", "SPCE", "WKHS",
    "QS", "BLNK", "AMC", "BYND", "TLRY", "CLOV", "PTON",
    "W", "OPEN", "SKLZ", "LMND", "RIG", "CLF", "AAL",
}


def read_zip(zip_path: str) -> pd.DataFrame:
    """Read all CSV files inside a zip archive into a single DataFrame."""
    frames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {zip_path}")
        for csv_name in csv_files:
            print(f"  Reading: {csv_name}")
            with zf.open(csv_name) as f:
                df = pd.read_csv(io.TextIOWrapper(f))
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all filtering criteria and return one row per ticker-year."""

    print(f"\nRaw data: {len(df):,} rows")

    # ── 0. Standardise column names ──
    df.columns = df.columns.str.strip().str.lower()

    # ── 1. Keep only target tickers ──
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].isin(TARGET_TICKERS)]
    print(f"After ticker filter: {len(df):,} rows ({df['ticker'].nunique()} tickers)")

    if df.empty:
        print("WARNING: No matching tickers found. Check the ticker column in your CSV.")
        return pd.DataFrame()

    # ── 2. Parse dates ──
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    df["year"] = df["date"].dt.year

    # ── 3. Keep only last 5 trading days of December (year-end) ──
    df_dec = df[df["date"].dt.month == 12]
    dec_dates = (
        df_dec.groupby("year")["date"]
        .apply(lambda s: s.drop_duplicates().nlargest(5))
        .reset_index(drop=True)
        .unique()
    )
    df = df[df["date"].isin(dec_dates)]
    print(f"After year-end filter (Dec last 5 days): {len(df):,} rows")

    # ── 4. Keep only Calls ──
    df["cp_flag"] = df["cp_flag"].astype(str).str.strip().str.upper()
    df = df[df["cp_flag"] == "C"]
    print(f"After Call filter: {len(df):,} rows")

    # ── 5. Compute days to expiry, keep 250-450 days ──
    df["days"] = (df["exdate"] - df["date"]).dt.days
    df = df[(df["days"] >= 250) & (df["days"] <= 450)]
    print(f"After expiry 250-450 days filter: {len(df):,} rows")

    # ── 6. ATM filter: delta between 0.40 and 0.60 ──
    df = df[df["delta"].notna() & (df["delta"] >= 0.40) & (df["delta"] <= 0.60)]
    print(f"After delta 0.40-0.60 filter: {len(df):,} rows")

    # ── 7. Implied vol quality filter ──
    df = df[
        df["impl_volatility"].notna()
        & (df["impl_volatility"] > 0)
        & (df["impl_volatility"] < 3.0)
    ]
    print(f"After impl_vol quality filter: {len(df):,} rows")

    # ── 8. Remove options with no trading activity ──
    if "volume" in df.columns and "open_interest" in df.columns:
        df = df[~((df["volume"] == 0) & (df["open_interest"] == 0))]
        print(f"After removing untradeable options: {len(df):,} rows")

    # ── 9. Convert strike_price (OptionMetrics stores strike * 1000) ──
    df["strike_price_actual"] = df["strike_price"] / 1000.0

    # ── 10. Pick best row per ticker-year ──
    #   Priority: delta closest to 0.50, then days closest to 365
    df["delta_dist"] = (df["delta"] - 0.50).abs()
    df["days_dist"] = (df["days"] - 365).abs()
    df = df.sort_values(["delta_dist", "days_dist"])
    result = df.groupby(["ticker", "year"], as_index=False).first()

    print(f"\nFinal result: {len(result)} rows ({result['ticker'].nunique()} tickers)")
    print(f"Years covered: {sorted(result['year'].unique())}")

    # ── 11. Select output columns ──
    out_cols = [
        "ticker", "date", "exdate", "days", "delta",
        "impl_volatility", "strike_price_actual", "volume",
    ]
    out = result[[c for c in out_cols if c in result.columns]].copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    return out


def merge_all():
    """Merge iv_2021.csv ~ iv_2025.csv into data/implied_vol_optionmetrics.csv."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    frames = []
    for year in range(2021, 2026):
        path = os.path.join(data_dir, f"iv_{year}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded {path}: {len(df)} rows")
            frames.append(df)
        else:
            print(f"  Not found: {path} (skipping)")

    if not frames:
        print("ERROR: No iv_YYYY.csv files found in data/")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    out_path = os.path.join(data_dir, "implied_vol_optionmetrics.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nMerged {len(merged)} rows -> {out_path}")
    print(f"Tickers: {merged['ticker'].nunique()}")
    print(f"\nPreview:")
    print(merged.head(10).to_string(index=False))


def main():
    # --merge mode: combine all yearly files
    if len(sys.argv) == 2 and sys.argv[1] == "--merge":
        print("=" * 60)
        print("Merging yearly IV files into implied_vol_optionmetrics.csv")
        print("=" * 60)
        merge_all()
        return

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python scripts/filter_optionmetrics.py <zip_path> <output_csv>")
        print("  python scripts/filter_optionmetrics.py --merge")
        print()
        print("Examples:")
        print("  python scripts/filter_optionmetrics.py ~/Downloads/opprcd_2021.zip data/iv_2021.csv")
        print("  python scripts/filter_optionmetrics.py ~/Downloads/opprcd_2022.zip data/iv_2022.csv")
        print()
        print("After all members finish, run:")
        print("  python scripts/filter_optionmetrics.py --merge")
        sys.exit(1)

    zip_path = sys.argv[1]
    out_path = sys.argv[2]

    print("=" * 60)
    print("OptionMetrics ATM Implied Volatility Filter")
    print(f"Input:  {zip_path}")
    print(f"Output: {out_path}")
    print("=" * 60)

    # Read
    print("\n[1/2] Reading zip file ...")
    df = read_zip(zip_path)

    # Filter
    print("\n[2/2] Filtering data ...")
    result = filter_data(df)

    if result.empty:
        print("\nWARNING: No data after filtering. Output file not created.")
        sys.exit(1)

    # Save
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"\nPreview (first 10 rows):")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
