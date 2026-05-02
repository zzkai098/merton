"""Fetch risk-free rate from FRED (with yfinance fallback)."""

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


def fetch_risk_free_rate(start: str, end: str) -> pd.DataFrame:
    """Return a daily risk-free rate series.

    Primary source is the FRED DGS1 (1-Year Treasury Constant Maturity Rate).
    If ``fredapi`` is unavailable or the API key is a placeholder, falls back
    to the ^TNX index (10-Year Treasury yield) via yfinance.

    Parameters
    ----------
    start, end : str
        Date strings in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Columns: date, risk_free_rate (as decimal, e.g. 0.05 for 5 %).
    """
    df = _try_fred(start, end)
    if df is not None:
        return df

    logger.info("FRED unavailable — falling back to yfinance ^TNX.")
    return _fallback_yfinance(start, end)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _try_fred(start: str, end: str) -> pd.DataFrame | None:
    """Attempt to fetch risk-free rate from FRED via ``fredapi``."""
    api_key = getattr(config, "FRED_API_KEY", None)
    if not api_key or "YOUR_" in api_key.upper():
        logger.info("FRED API key is missing or placeholder — skipping FRED.")
        return None

    try:
        from fredapi import Fred  # type: ignore[import-untyped]

        fred = Fred(api_key=api_key)
        series = fred.get_series(
            config.RISK_FREE_SERIES,
            observation_start=start,
            observation_end=end,
        )
    except Exception as exc:
        logger.warning("fredapi request failed: %s", exc)
        return None

    if series is None or series.empty:
        logger.warning("FRED returned empty series for %s.", config.RISK_FREE_SERIES)
        return None

    df = series.to_frame(name="risk_free_rate").reset_index()
    df.columns = ["date", "risk_free_rate"]
    # FRED reports rates as percentages (e.g. 5.0 for 5 %) — convert to decimal
    df["risk_free_rate"] = df["risk_free_rate"] / 100.0
    df["risk_free_rate"] = df["risk_free_rate"].ffill()
    df = df.dropna(subset=["risk_free_rate"])
    logger.info("FRED risk-free rate: %d observations.", len(df))
    return df


def _fallback_yfinance(start: str, end: str) -> pd.DataFrame:
    """Fetch ^TNX (10-Year Treasury yield) via yfinance as a fallback."""
    import yfinance as yf

    try:
        raw = yf.download("^TNX", start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:
        logger.error("yfinance ^TNX download failed: %s", exc)
        return pd.DataFrame(columns=["date", "risk_free_rate"])

    if raw is None or raw.empty:
        logger.warning("yfinance returned no data for ^TNX.")
        return pd.DataFrame(columns=["date", "risk_free_rate"])

    df = raw[["Close"]].copy()
    df.columns = ["risk_free_rate"]
    # ^TNX quotes the yield in percentage points (e.g. 4.5 → 0.045)
    df["risk_free_rate"] = df["risk_free_rate"] / 100.0
    df["risk_free_rate"] = df["risk_free_rate"].ffill()
    df = df.dropna(subset=["risk_free_rate"])
    df = df.reset_index()
    df.columns = ["date", "risk_free_rate"]
    logger.info("yfinance ^TNX risk-free rate: %d observations.", len(df))
    return df
