"""Load implied volatility from OptionMetrics CSV file."""

import logging
import os

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Expected CSV path
_DEFAULT_IV_PATH = os.path.join("data_raw", "implied_vol_optionmetrics.csv")


def load_implied_vol(path: str | None = None) -> pd.DataFrame:
    """Read OptionMetrics implied vol CSV and return one row per ticker-year.

    The CSV should contain at least: ``ticker``, ``date``, ``impl_volatility``.
    Optionally ``days`` (to expiry) and ``delta`` for finer filtering.

    Parameters
    ----------
    path : str | None
        Path to the CSV file.  Defaults to ``data_raw/implied_vol_optionmetrics.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``year``, ``implied_vol``.
    """
    path = path or _DEFAULT_IV_PATH
    if not os.path.exists(path):
        logger.warning("Implied vol file not found: %s — skipping.", path)
        return pd.DataFrame(columns=["ticker", "year", "implied_vol"])

    df = pd.read_csv(path)
    logger.info("Loaded implied vol CSV: %d rows from %s", len(df), path)

    # Normalise column names (handle various OptionMetrics exports)
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("impl_volatility", "implied_volatility", "impl_vol", "iv"):
            col_map[col] = "impl_volatility"
        elif lc == "ticker":
            col_map[col] = "ticker"
        elif lc == "date":
            col_map[col] = "date"
        elif lc == "days":
            col_map[col] = "days"
        elif lc == "delta":
            col_map[col] = "delta"
    df = df.rename(columns=col_map)

    if "ticker" not in df.columns or "impl_volatility" not in df.columns:
        logger.error("CSV must contain 'ticker' and 'impl_volatility' columns.")
        return pd.DataFrame(columns=["ticker", "year", "implied_vol"])

    # Parse date → year
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Filter: valid implied vol
    df = df[df["impl_volatility"].notna() & (df["impl_volatility"] > 0) & (df["impl_volatility"] < 3.0)]

    # If multiple rows per ticker-year, pick the one closest to 365 days
    if "days" in df.columns:
        df["_dist"] = (df["days"] - 365).abs()
        df = df.sort_values("_dist").groupby(["ticker", "year"], as_index=False).first()
        df = df.drop(columns=["_dist"])

    result = df[["ticker", "year"]].copy()
    result["implied_vol"] = df["impl_volatility"].values

    logger.info(
        "Implied vol: %d ticker-year observations, %d unique tickers.",
        len(result), result["ticker"].nunique(),
    )
    return result
