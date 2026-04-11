"""Fetch equity data from Yahoo Finance."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


def fetch_equity_data(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download daily adjusted close prices and compute market cap / equity vol.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols.
    start, end : str
        Date strings in ``YYYY-MM-DD`` format defining the download window.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, date, price, market_cap, equity_vol
    """
    # ------------------------------------------------------------------
    # 1. Batch-download adjusted close prices
    # ------------------------------------------------------------------
    logger.info("Downloading equity prices for %d tickers …", len(tickers))
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.error("yfinance batch download failed: %s", exc)
        return pd.DataFrame(
            columns=["ticker", "date", "price", "market_cap", "equity_vol"]
        )

    # yf.download returns multi-level columns when len(tickers) > 1.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — wrap into a single-column DataFrame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # ------------------------------------------------------------------
    # 2. Gather shares outstanding per ticker (for market-cap calc)
    # ------------------------------------------------------------------
    shares_outstanding: dict[str, Optional[float]] = {}
    latest_market_cap: dict[str, Optional[float]] = {}

    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info
            shares = info.get("sharesOutstanding")
            mkt_cap = info.get("marketCap")
            shares_outstanding[tkr] = float(shares) if shares else None
            latest_market_cap[tkr] = float(mkt_cap) if mkt_cap else None
        except Exception as exc:
            logger.warning("Could not retrieve info for %s: %s", tkr, exc)
            shares_outstanding[tkr] = None
            latest_market_cap[tkr] = None

    # ------------------------------------------------------------------
    # 3. Build per-ticker time series
    # ------------------------------------------------------------------
    frames: list[pd.DataFrame] = []

    for tkr in tickers:
        if tkr not in prices.columns:
            logger.warning("No price data for %s — skipping.", tkr)
            continue

        series = prices[tkr].dropna()
        if series.empty:
            logger.warning("Empty price series for %s — skipping.", tkr)
            continue

        df = pd.DataFrame({"price": series})
        df.index.name = "date"

        # Market cap
        so = shares_outstanding.get(tkr)
        if so is not None:
            df["market_cap"] = df["price"] * so
        elif latest_market_cap.get(tkr) is not None:
            # Fallback: use the latest reported market cap as a constant
            logger.warning(
                "%s: sharesOutstanding unavailable — using latest marketCap as fallback.",
                tkr,
            )
            df["market_cap"] = latest_market_cap[tkr]
        else:
            logger.warning(
                "%s: no market-cap information available — column will be NaN.",
                tkr,
            )
            df["market_cap"] = np.nan

        # Annualised equity volatility (rolling 252-day)
        log_ret = np.log(df["price"] / df["price"].shift(1))
        df["equity_vol"] = (
            log_ret.rolling(window=config.VOL_WINDOW, min_periods=config.VOL_WINDOW)
            .std()
            .multiply(np.sqrt(config.TRADING_DAYS))
        )

        df["ticker"] = tkr
        df = df.reset_index()
        frames.append(df)

    if not frames:
        logger.warning("No equity data collected for any ticker.")
        return pd.DataFrame(
            columns=["ticker", "date", "price", "market_cap", "equity_vol"]
        )

    result = pd.concat(frames, ignore_index=True)
    result = result[["ticker", "date", "price", "market_cap", "equity_vol"]]
    logger.info("Equity data: %d rows for %d tickers.", len(result), result["ticker"].nunique())
    return result
