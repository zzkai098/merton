"""
OptionMetrics Option Data Filter
=================================
Each team member runs this script on their assigned year's zip/csv file
to extract ATM call implied volatility for our 163 target tickers.

Usage:
    python scripts/filter_optionmetrics.py  <zip_or_csv_path>  <output_csv_path>

Examples:
    python scripts/filter_optionmetrics.py  ~/Downloads/opprcd_2021.zip  data_raw/iv_2021.csv
    python scripts/filter_optionmetrics.py  ~/Downloads/2025.csv         data_raw/iv_2025.csv

After all 5 members finish, combine the outputs:
    python scripts/filter_optionmetrics.py --merge

Filtering logic:
    1. Keep only our 163 target tickers
    2. Keep only the last 5 trading days of December each year (year-end)
       — if no December data exists, fall back to the last 5 available dates
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
    - Supports both .zip and .csv input files.
    - Large files are read in chunks to avoid memory issues.
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

CHUNK_SIZE = 1_000_000


def read_and_filter_csv(csv_path_or_fh):
    """Read a CSV in chunks, applying early filters to keep memory low."""
    # Columns we need (read only these to save memory)
    use_cols = [
        "ticker", "date", "exdate", "cp_flag", "delta",
        "impl_volatility", "strike_price", "volume", "open_interest",
    ]

    reader = pd.read_csv(
        csv_path_or_fh,
        chunksize=CHUNK_SIZE,
        usecols=lambda c: c.strip().lower() in {x.lower() for x in use_cols},
        low_memory=False,
    )

    filtered_chunks = []
    total_rows = 0
    kept_rows = 0

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        chunk.columns = chunk.columns.str.strip().str.lower()

        # 1. Ticker filter (drop >99% of rows immediately)
        chunk["ticker"] = chunk["ticker"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["ticker"].isin(TARGET_TICKERS)]
        if chunk.empty:
            if (i + 1) % 50 == 0:
                print(f"  Chunk {i+1}: {total_rows:,} rows read, {kept_rows:,} kept", flush=True)
            continue

        # 2. Call only
        chunk = chunk.copy()
        chunk["cp_flag"] = chunk["cp_flag"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["cp_flag"] == "C"]
        if chunk.empty:
            continue

        # 3. Parse dates & compute days to expiry
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["exdate"] = pd.to_datetime(chunk["exdate"], errors="coerce")
        chunk["days"] = (chunk["exdate"] - chunk["date"]).dt.days

        # 4. Keep 250-450 days to expiry
        chunk = chunk[(chunk["days"] >= 250) & (chunk["days"] <= 450)]
        if chunk.empty:
            continue

        # 5. ATM: delta 0.40 - 0.60
        chunk["delta"] = pd.to_numeric(chunk["delta"], errors="coerce")
        chunk = chunk[chunk["delta"].notna() & (chunk["delta"] >= 0.40) & (chunk["delta"] <= 0.60)]
        if chunk.empty:
            continue

        # 6. Implied vol quality
        chunk["impl_volatility"] = pd.to_numeric(chunk["impl_volatility"], errors="coerce")
        chunk = chunk[
            chunk["impl_volatility"].notna()
            & (chunk["impl_volatility"] > 0)
            & (chunk["impl_volatility"] < 3.0)
        ]
        if chunk.empty:
            continue

        # 7. Remove no-trading options
        if "volume" in chunk.columns and "open_interest" in chunk.columns:
            chunk["volume"] = pd.to_numeric(chunk["volume"], errors="coerce").fillna(0)
            chunk["open_interest"] = pd.to_numeric(chunk["open_interest"], errors="coerce").fillna(0)
            chunk = chunk[~((chunk["volume"] == 0) & (chunk["open_interest"] == 0))]
        if chunk.empty:
            continue

        kept_rows += len(chunk)
        filtered_chunks.append(chunk)

        if (i + 1) % 50 == 0:
            print(f"  Chunk {i+1}: {total_rows:,} rows read, {kept_rows:,} kept", flush=True)

    print(f"  Done: {total_rows:,} total rows, {kept_rows:,} after filters", flush=True)

    if not filtered_chunks:
        return pd.DataFrame()
    return pd.concat(filtered_chunks, ignore_index=True)


def read_input(file_path: str) -> pd.DataFrame:
    """Read from zip or csv, with chunked filtering."""
    if file_path.lower().endswith(".zip"):
        # For zip: extract CSV inside, then chunk-read it
        frames = []
        with zipfile.ZipFile(file_path, "r") as zf:
            csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in: {file_path}")
            for csv_name in csv_files:
                print(f"  Reading from zip: {csv_name}", flush=True)
                with zf.open(csv_name) as f:
                    df = read_and_filter_csv(io.TextIOWrapper(f))
                    if not df.empty:
                        frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    else:
        # Plain CSV
        print(f"  Reading CSV: {file_path}", flush=True)
        return read_and_filter_csv(file_path)


def pick_best_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pick one best row per ticker-year from pre-filtered data."""
    if df.empty:
        return pd.DataFrame()

    df["year"] = df["date"].dt.year
    print(f"\nPre-filtered data: {len(df):,} rows, {df['ticker'].nunique()} tickers", flush=True)
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}", flush=True)
    print(f"Years: {sorted(df['year'].unique())}", flush=True)

    results = []
    for year, year_grp in df.groupby("year"):
        # Try December last 5 trading days first
        dec_data = year_grp[year_grp["date"].dt.month == 12]
        if not dec_data.empty:
            dec_dates = sorted(dec_data["date"].unique())
            target_dates = set(dec_dates[-5:])
            subset = dec_data[dec_data["date"].isin(target_dates)]
            print(f"  Year {year}: using Dec last 5 days", flush=True)
        else:
            # Fallback: use the last 5 available trading dates
            all_dates = sorted(year_grp["date"].unique())
            target_dates = set(all_dates[-5:])
            subset = year_grp[year_grp["date"].isin(target_dates)]
            date_strs = sorted(d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in target_dates)
            print(f"  Year {year}: no Dec data, using last 5 dates: {date_strs}", flush=True)

        # Per ticker: pick latest date, then best delta/days
        for ticker, t_grp in subset.groupby("ticker"):
            latest_date = t_grp["date"].max()
            day_grp = t_grp[t_grp["date"] == latest_date].copy()
            day_grp["delta_dist"] = (day_grp["delta"] - 0.50).abs()
            day_grp["days_dist"] = (day_grp["days"] - 365).abs()
            day_grp = day_grp.sort_values(["delta_dist", "days_dist"])
            results.append(day_grp.iloc[0])

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(results)
    print(f"\nFinal: {len(result)} rows ({result['ticker'].nunique()} tickers)", flush=True)
    print(f"Years covered: {sorted(result['year'].unique())}", flush=True)

    # Convert strike_price (OptionMetrics stores strike * 1000)
    if "strike_price" in result.columns:
        result["strike_price_actual"] = result["strike_price"] / 1000.0

    # Select output columns
    out_cols = ["ticker", "date", "days", "delta", "impl_volatility"]
    out = result[[c for c in out_cols if c in result.columns]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def merge_all():
    """Merge iv_2021.csv ~ iv_2025.csv into data_raw/implied_vol_optionmetrics.csv."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data_raw")
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
        print("ERROR: No iv_YYYY.csv files found in data_raw/")
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
        print("  python scripts/filter_optionmetrics.py <zip_or_csv_path> <output_csv>")
        print("  python scripts/filter_optionmetrics.py --merge")
        print()
        print("Examples:")
        print("  python scripts/filter_optionmetrics.py ~/Downloads/opprcd_2021.zip data_raw/iv_2021.csv")
        print("  python scripts/filter_optionmetrics.py ~/Downloads/2025.csv        data_raw/iv_2025.csv")
        print()
        print("After all members finish, run:")
        print("  python scripts/filter_optionmetrics.py --merge")
        sys.exit(1)

    file_path = sys.argv[1]
    out_path = sys.argv[2]

    print("=" * 60, flush=True)
    print("OptionMetrics ATM Implied Volatility Filter", flush=True)
    print(f"Input:  {file_path}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print("=" * 60, flush=True)

    # Read & pre-filter
    print("\n[1/2] Reading and pre-filtering ...", flush=True)
    df = read_input(file_path)

    if df.empty:
        print("\nWARNING: No data after filtering. Output file not created.")
        sys.exit(1)

    # Pick best rows
    print("\n[2/2] Selecting best row per ticker-year ...", flush=True)
    result = pick_best_rows(df)

    if result.empty:
        print("\nWARNING: No data after selection. Output file not created.")
        sys.exit(1)

    # Save
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)
    print(f"\nPreview (first 20 rows):", flush=True)
    print(result.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
