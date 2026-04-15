"""Merge all data sources into a firm-level panel dataset."""

import logging
import os

import numpy as np
import pandas as pd

import config
from data.equity import fetch_equity_data
from data.balance_sheet import fetch_balance_sheet
from data.risk_free import fetch_risk_free_rate
from data.implied_vol import load_implied_vol

logger = logging.getLogger(__name__)


def build_panel(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Construct an annual firm-level panel suitable for credit-risk modelling.

    Parameters
    ----------
    tickers : list[str] | None
        Ticker symbols.  Defaults to ``config.ALL_TICKERS``.
    start, end : str | None
        Date window.  Defaults to ``config.START_DATE`` / ``config.END_DATE``.

    Returns
    -------
    pd.DataFrame
        Merged panel saved to ``output/panel_data.csv`` and returned.
    """
    tickers = tickers or config.ALL_TICKERS
    start = start or config.START_DATE
    end = end or config.END_DATE

    # ------------------------------------------------------------------
    # 1. Fetch raw data
    # ------------------------------------------------------------------
    logger.info("Fetching equity data …")
    equity_df = fetch_equity_data(tickers, start, end)

    logger.info("Fetching balance-sheet data …")
    bs_df = fetch_balance_sheet(tickers)

    logger.info("Fetching risk-free rate …")
    rf_df = fetch_risk_free_rate(start, end)

    # ------------------------------------------------------------------
    # 2. Equity: take last available observation per ticker per year
    # ------------------------------------------------------------------
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    equity_df["year"] = equity_df["date"].dt.year
    equity_annual = (
        equity_df.sort_values("date")
        .groupby(["ticker", "year"], as_index=False)
        .last()
    )

    # ------------------------------------------------------------------
    # 2b. Overlay implied volatility from OptionMetrics (if enabled)
    # ------------------------------------------------------------------
    if getattr(config, "USE_IMPLIED_VOL", False):
        logger.info("Loading implied volatility from OptionMetrics CSV …")
        iv_df = load_implied_vol()
        if not iv_df.empty:
            equity_annual = equity_annual.merge(
                iv_df[["ticker", "year", "implied_vol"]],
                on=["ticker", "year"],
                how="left",
            )
            # Use implied vol where available; fall back to historical vol
            equity_annual["equity_vol"] = (
                equity_annual["implied_vol"]
                .fillna(equity_annual["equity_vol"])
            )
            n_total = len(equity_annual)
            n_implied = equity_annual["implied_vol"].notna().sum()
            logger.info(
                "Implied vol applied to %d / %d firm-year rows. Rest use historical vol.",
                n_implied, n_total,
            )
            equity_annual = equity_annual.drop(columns=["implied_vol"])

    # ------------------------------------------------------------------
    # 3. Balance sheet: assign fiscal year
    # ------------------------------------------------------------------
    bs_df["date"] = pd.to_datetime(bs_df["date"])
    bs_df["year"] = bs_df["date"].dt.year

    bs_cols = [
        "ticker",
        "year",
        "total_debt",
        "short_term_debt",
        "long_term_debt",
        "default_point",
        "total_assets",
    ]
    bs_annual = bs_df[bs_cols].copy()

    # ------------------------------------------------------------------
    # 4. Risk-free rate: annual average
    # ------------------------------------------------------------------
    rf_df["date"] = pd.to_datetime(rf_df["date"])
    rf_df["year"] = rf_df["date"].dt.year
    rf_annual = rf_df.groupby("year", as_index=False)["risk_free_rate"].mean()

    # ------------------------------------------------------------------
    # 5. Merge
    # ------------------------------------------------------------------
    panel = equity_annual.merge(bs_annual, on=["ticker", "year"], how="left")
    panel = panel.merge(rf_annual, on="year", how="left")

    # ------------------------------------------------------------------
    # 6. Derived columns
    # ------------------------------------------------------------------
    panel["default_label"] = panel["ticker"].isin(config.DEFAULT_TICKERS).astype(int)

    panel["leverage_ratio"] = panel["total_debt"] / (
        panel["market_cap"] + panel["total_debt"]
    )

    panel["log_market_cap"] = np.log(panel["market_cap"])

    # ------------------------------------------------------------------
    # 7. Drop rows missing critical fields
    # ------------------------------------------------------------------
    critical = ["market_cap", "total_debt", "equity_vol"]
    before = len(panel)
    panel = panel.dropna(subset=critical).reset_index(drop=True)
    dropped = before - len(panel)
    if dropped:
        logger.info("Dropped %d rows with missing critical values.", dropped)

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, "panel_data.csv")
    panel.to_csv(out_path, index=False)
    logger.info("Panel saved to %s — %d rows, %d tickers.", out_path, len(panel), panel["ticker"].nunique())

    return panel
