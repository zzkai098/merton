"""
Markit CDS Data Filter for Hull-White Model
============================================
Filters CDS parspread data for 163 target tickers, 2021-2025 year-end.

Usage:
    python scripts/filter_cds.py

    Place your zip files in ~/Downloads/ (or edit ZIP_PATHS below):
        - 2020-2022.zip
        - 2023-2024.zip
        - 2025(1).zip

Output:
    data_raw/cds_parspread.csv  — one row per ticker-year, ~815 rows

Filtering logic:
    1. Keep only 163 target tickers
    2. Keep only years 2021-2025
    3. Keep only last trading day of December each year (year-end)
    4. Keep only tenor = '5Y' (most liquid; fallback from 1Y if needed)
    5. Keep only tier = 'SNRFOR' (Senior Foreign, most standard)
    6. Keep only currency = 'USD'
    7. Keep only region = 'North America'
    8. Keep only parspread > 0 and not null
    9. One row per ticker-year
"""

import os
import sys
import zipfile
import io
import pandas as pd

# ── Zip file paths — edit if your files are not in ~/Downloads ──
ZIP_PATHS = [
    os.path.expanduser("~/Downloads/2020-2022.zip"),
    os.path.expanduser("~/Downloads/2023-2024.zip"),
    os.path.expanduser("~/Downloads/2025(1).zip"),
]

# ── Output path ──
OUTPUT_PATH = "data_raw/cds_parspread.csv"

# ── Target years ──
TARGET_YEARS = {2021, 2022, 2023, 2024, 2025}

# ── 163 target tickers ──
TARGET_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "CRM", "ADBE",
    "AMD", "QCOM", "AVGO", "TXN", "NFLX", "PYPL", "INTC", "CSCO",
    "ORCL", "IBM", "NOW", "INTU", "AMAT", "MU", "LRCX", "KLAC",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BK", "AXP",
    "SCHW", "BLK", "SPGI", "USB", "PNC", "TFC", "MET", "AIG",
    "PRU", "ALL", "TRV", "CME",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT",
    "BMY", "GILD", "AMGN", "MDT", "CI", "CVS", "HUM", "BSX",
    "SYK", "ZTS", "REGN", "VRTX",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
    "OXY", "HAL", "DVN", "WMB", "KMI",
    "HD", "LOW", "TGT", "COST", "NKE", "SBUX", "MCD", "YUM",
    "GM", "F", "TSLA", "LULU", "ROST", "TJX", "DHI", "LEN",
    "WMT", "PG", "KO", "PEP", "CL", "KHC", "GIS", "SJM",
    "HSY", "MO", "PM", "STZ", "KR", "SYY",
    "BA", "CAT", "GE", "MMM", "HON", "RTX", "LMT", "NOC",
    "DE", "EMR", "ITW", "WM", "RSG", "FDX", "UPS", "CSX",
    "UNP", "NSC",
    "T", "VZ", "CMCSA", "CHTR", "DIS", "WBD", "FOX",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE",
    "DAL", "UAL", "LUV",
    "HTZ", "PLUG", "FCEL", "LCID", "RIVN", "SPCE", "WKHS",
    "QS", "BLNK", "AMC", "BYND", "TLRY", "CLOV", "PTON",
    "W", "OPEN", "SKLZ", "LMND", "RIG", "CLF", "AAL",
}

DISTRESSED = {
    "HTZ", "PLUG", "FCEL", "LCID", "RIVN", "SPCE", "WKHS",
    "QS", "BLNK", "AMC", "BYND", "TLRY", "CLOV", "PTON",
    "W", "OPEN", "SKLZ", "LMND", "RIG", "CLF", "AAL"
}


def read_zip_chunked(zip_path: str) -> pd.DataFrame:
    """Read zip in chunks, pre-filtering ticker and year to save memory."""
    if not os.path.exists(zip_path):
        print(f"  WARNING: File not found, skipping: {zip_path}")
        return pd.DataFrame()

    frames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            print(f"  WARNING: No CSV files found in {zip_path}")
            return pd.DataFrame()

        for csv_name in csv_files:
            print(f"  Reading: {csv_name}")
            with zf.open(csv_name) as f:
                reader = pd.read_csv(
                    io.TextIOWrapper(f),
                    chunksize=500_000,
                    low_memory=False
                )
                for i, chunk in enumerate(reader):
                    # Standardise column names
                    chunk.columns = chunk.columns.str.strip().str.lower()

                    # Pre-filter ticker
                    if "ticker" in chunk.columns:
                        chunk["ticker"] = chunk["ticker"].astype(str).str.strip().str.upper()
                        chunk = chunk[chunk["ticker"].isin(TARGET_TICKERS)]

                    # Pre-filter year
                    if not chunk.empty and "date" in chunk.columns:
                        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                        chunk = chunk[chunk["date"].dt.year.isin(TARGET_YEARS)]

                    # Pre-filter December only
                    if not chunk.empty:
                        chunk = chunk[chunk["date"].dt.month == 12]

                    if not chunk.empty:
                        frames.append(chunk)

                    print(f"    chunk {i+1} done", end="\r")
    print()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def filter_cds(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all CDS filtering criteria, return one row per ticker-year."""

    print(f"\nData after chunked read: {len(df):,} rows")

    # Standardise columns
    df.columns = df.columns.str.strip().str.lower()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # ── 1. Tenor filter: try 5Y first ──
    df["tenor"] = df["tenor"].astype(str).str.strip().str.upper()
    df_5y = df[df["tenor"] == "5Y"]
    print(f"After tenor=5Y filter: {len(df_5y):,} rows ({df_5y['ticker'].nunique()} tickers)")

    # Check if 1Y is available as alternative
    df_1y = df[df["tenor"] == "1Y"]
    print(f"For reference, tenor=1Y has: {len(df_1y):,} rows ({df_1y['ticker'].nunique()} tickers)")

    # Use 5Y as primary
    df = df_5y.copy()

    # ── 2. Tier filter ──
    df["tier"] = df["tier"].astype(str).str.strip().str.upper()
    df = df[df["tier"] == "SNRFOR"]
    print(f"After tier=SNRFOR filter: {len(df):,} rows")

    # ── 3. Currency filter ──
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    df = df[df["currency"] == "USD"]
    print(f"After currency=USD filter: {len(df):,} rows")

    # ── 4. Region filter ──
    if "region" in df.columns:
        df["region"] = df["region"].astype(str).str.strip()
        df = df[df["region"].str.contains("N.Amer", case=False, na=False)]
        print(f"After region=North America filter: {len(df):,} rows")

    # ── 5. parspread quality filter ──
    df = df[df["parspread"].notna() & (df["parspread"] > 0)]
    print(f"After parspread > 0 filter: {len(df):,} rows")

    # ── 6. Keep only last trading day of December each year ──
    dec_dates = (
        df.groupby("year")["date"]
        .apply(lambda s: s.drop_duplicates().nlargest(1))
        .reset_index(drop=True)
        .unique()
    )
    df = df[df["date"].isin(dec_dates)]
    print(f"After year-end (last Dec day) filter: {len(df):,} rows")

    # ── 7. One row per ticker-year ──
    # If duplicates remain, keep the one with highest compositedepth5y (most liquid)
    if "compositedepth5y" in df.columns:
        df = df.sort_values("compositedepth5y", ascending=False)
    result = df.groupby(["ticker", "year"], as_index=False).first()

    print(f"\nFinal result: {len(result)} rows ({result['ticker'].nunique()} tickers)")
    print(f"Years covered: {sorted(result['year'].unique())}")

    # ── 8. Select output columns ──
    out_cols = ["ticker", "date", "year", "tenor", "tier", "currency",
                "parspread", "dp", "avrating", "impliedrating"]
    out = result[[c for c in out_cols if c in result.columns]].copy()
    out = out.sort_values(["ticker", "year"]).reset_index(drop=True)

    return out


def check_coverage(result: pd.DataFrame):
    """Print coverage report, especially for distressed tickers."""
    print("\n" + "=" * 50)
    print("COVERAGE REPORT")
    print("=" * 50)

    for year in sorted(result["year"].unique()):
        yr_tickers = set(result[result["year"] == year]["ticker"])
        missing = TARGET_TICKERS - yr_tickers
        missing_distressed = DISTRESSED - yr_tickers
        print(f"\n{year}: {len(yr_tickers)}/163 tickers covered, "
              f"{len(missing)} missing")
        if missing_distressed:
            print(f"  ⚠ Missing distressed: {sorted(missing_distressed)}")
        else:
            print(f"  ✓ All distressed tickers present")

    print()


def main():
    print("=" * 60)
    print("Markit CDS Filter — Hull-White Model")
    print("=" * 60)

    # Read all zips
    all_frames = []
    for zip_path in ZIP_PATHS:
        print(f"\n[Reading] {zip_path}")
        df = read_zip_chunked(zip_path)
        if not df.empty:
            all_frames.append(df)
            print(f"  -> {len(df):,} rows loaded")

    if not all_frames:
        print("\nERROR: No data loaded from any zip file.")
        sys.exit(1)

    df_all = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows across all zips: {len(df_all):,}")

    # Filter
    print("\n[Filtering]")
    result = filter_cds(df_all)

    if result.empty:
        print("\nWARNING: No data after filtering.")
        sys.exit(1)

    # Coverage report
    check_coverage(result)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"\nPreview (first 10 rows):")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
