"""Fetch balance sheet data from Yahoo Finance."""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Common field-name variants across different yfinance versions / locales.
_TOTAL_DEBT_KEYS = ["Total Debt", "TotalDebt", "totalDebt"]
_SHORT_TERM_DEBT_KEYS = [
    "Current Debt",
    "CurrentDebt",
    "Short Long Term Debt",
    "ShortLongTermDebt",
    "Current Debt And Capital Lease Obligation",
    "Short Term Debt",
]
_LONG_TERM_DEBT_KEYS = [
    "Long Term Debt",
    "LongTermDebt",
    "Long Term Debt And Capital Lease Obligation",
]
_TOTAL_ASSETS_KEYS = ["Total Assets", "TotalAssets", "totalAssets"]


def _first_match(index: pd.Index, candidates: list[str]) -> str | None:
    """Return the first candidate that exists in *index*, or ``None``."""
    for key in candidates:
        if key in index:
            return key
    return None


def fetch_balance_sheet(tickers: list[str]) -> pd.DataFrame:
    """Download annual balance-sheet items for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, date, total_debt, short_term_debt, long_term_debt,
        default_point, total_assets
    """
    frames: list[pd.DataFrame] = []

    for tkr in tickers:
        try:
            bs = yf.Ticker(tkr).balance_sheet
        except Exception as exc:
            logger.warning("Failed to fetch balance sheet for %s: %s", tkr, exc)
            continue

        if bs is None or bs.empty:
            logger.warning("No balance-sheet data for %s — skipping.", tkr)
            continue

        # bs: index = line items, columns = fiscal-period end dates
        idx = bs.index

        td_key = _first_match(idx, _TOTAL_DEBT_KEYS)
        st_key = _first_match(idx, _SHORT_TERM_DEBT_KEYS)
        lt_key = _first_match(idx, _LONG_TERM_DEBT_KEYS)
        ta_key = _first_match(idx, _TOTAL_ASSETS_KEYS)

        for col_date in bs.columns:
            total_debt = float(bs.at[td_key, col_date]) if td_key else np.nan
            short_term = float(bs.at[st_key, col_date]) if st_key else np.nan
            long_term = float(bs.at[lt_key, col_date]) if lt_key else np.nan
            total_assets = float(bs.at[ta_key, col_date]) if ta_key else np.nan

            # Infer missing debt components when possible
            if np.isnan(total_debt) and not (np.isnan(short_term) and np.isnan(long_term)):
                total_debt = np.nansum([short_term, long_term])
            # Guard: if total_debt ended up as 0 (e.g. both components were NaN),
            # treat it as missing so merge.py dropna() filters it out.
            if total_debt == 0:
                total_debt = np.nan
            if np.isnan(short_term) and not np.isnan(total_debt) and not np.isnan(long_term):
                short_term = total_debt - long_term
            if np.isnan(long_term) and not np.isnan(total_debt) and not np.isnan(short_term):
                long_term = total_debt - short_term

            # VK default point: STD + 0.5 * LTD
            if not np.isnan(short_term) and not np.isnan(long_term):
                default_point = short_term + 0.5 * long_term
            else:
                default_point = np.nan

            frames.append(
                {
                    "ticker": tkr,
                    "date": pd.Timestamp(col_date),
                    "total_debt": total_debt,
                    "short_term_debt": short_term,
                    "long_term_debt": long_term,
                    "default_point": default_point,
                    "total_assets": total_assets,
                }
            )

    if not frames:
        logger.warning("No balance-sheet data collected for any ticker.")
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "total_debt",
                "short_term_debt",
                "long_term_debt",
                "default_point",
                "total_assets",
            ]
        )

    result = pd.DataFrame(frames)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info(
        "Balance-sheet data: %d rows for %d tickers.",
        len(result),
        result["ticker"].nunique(),
    )
    return result
