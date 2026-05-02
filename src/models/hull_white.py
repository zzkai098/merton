"""
Hull-White (2000) reduced-form credit risk model — CDS bootstrap implementation.

A default intensity (hazard rate) is *bootstrapped* from observed CDS par
spreads under the standard reduced-form pricing equation.  No accounting
features are used.

Two estimators are provided for a flat (piecewise-constant, single-tenor)
hazard rate:

1. **Credit-triangle approximation** (closed form):

       lambda ≈ s / (1 - R)

2. **PV-equation numerical solver** (recommended):

       PV_protection(lambda) = PV_premium(lambda) * s

   Solved via Brent's method with accrual-on-default on a quarterly grid.

Recovery rate selection per row, in priority order:
    1. ``estimated_rr`` column — firm-specific RR from the balance sheet
       (total_assets / total_debt, clipped to [config.RR_LO, config.RR_HI]).
    2. Scalar ``recovery_rate`` argument.
    3. ``config.RECOVERY_RATE`` (0.40, the ISDA convention).
The actual RR used per row is written to ``hw_recovery_rate``.

Risk-free rate selection per row, in priority order:
    1. ``risk_free_rate`` column on the input panel.
    2. Scalar ``risk_free_rate`` argument.

Reference
---------
Hull, J. & White, A. (2000). Valuing credit default swaps I: No counterparty
default risk. *Journal of Derivatives*, 8(1), 29-40.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults (tenor / RR bounds live in config; solver bounds local)
# ---------------------------------------------------------------------------
_LAMBDA_LO = 1e-8
_LAMBDA_HI = 5.0


# ---------------------------------------------------------------------------
# Estimator 1: closed-form credit-triangle approximation
# ---------------------------------------------------------------------------

def hazard_credit_triangle(spread: float, recovery_rate: float) -> float:
    """Closed-form approximation: lambda = s / (1 - R)."""
    if not np.isfinite(spread) or not np.isfinite(recovery_rate):
        return np.nan
    if spread < 0 or recovery_rate < 0 or recovery_rate >= 1.0:
        return np.nan
    return float(spread / (1.0 - recovery_rate))


# ---------------------------------------------------------------------------
# Estimator 2: numerical solver from the CDS PV equation
# ---------------------------------------------------------------------------

def _cds_pv_residual(
    lam: float,
    spread: float,
    recovery_rate: float,
    tenor: float,
    risk_free_rate: float,
    payments_per_year: int,
) -> float:
    """PV(protection leg) - spread * PV(premium leg + accrual)."""
    dt = 1.0 / payments_per_year
    n  = int(round(tenor * payments_per_year))
    i  = np.arange(1, n + 1)
    t_i = i * dt
    m_i = (i - 0.5) * dt

    df_t = np.exp(-risk_free_rate * t_i)
    df_m = np.exp(-risk_free_rate * m_i)

    surv_t        = np.exp(-lam * t_i)
    surv_t_prev   = np.exp(-lam * (t_i - dt))
    p_default_i   = surv_t_prev - surv_t

    pv_premium = (
        np.sum(df_t * surv_t * dt)
        + np.sum(df_m * p_default_i * (dt / 2.0))
    )
    pv_protection = (1.0 - recovery_rate) * np.sum(df_m * p_default_i)

    return pv_protection - spread * pv_premium


def hazard_pv_solve(
    spread: float,
    recovery_rate: float,
    tenor: float | None = None,
    risk_free_rate: float = 0.0,
    payments_per_year: int | None = None,
) -> float:
    """Bootstrap a flat hazard rate from a single CDS par spread via Brent."""
    if tenor is None:
        tenor = config.CDS_TENOR
    if payments_per_year is None:
        payments_per_year = config.CDS_PAYMENTS_PER_YEAR
    if not np.isfinite(spread) or not np.isfinite(recovery_rate):
        return np.nan
    if spread <= 0 or recovery_rate < 0 or recovery_rate >= 1.0:
        return np.nan
    if tenor <= 0 or payments_per_year < 1:
        return np.nan

    args = (spread, recovery_rate, tenor, risk_free_rate, payments_per_year)
    f_lo = _cds_pv_residual(_LAMBDA_LO, *args)
    f_hi = _cds_pv_residual(_LAMBDA_HI, *args)

    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return np.nan
    if f_lo * f_hi > 0:
        logger.warning(
            "PV residual no sign change for spread=%.6f, R=%.2f; "
            "falling back to credit-triangle.",
            spread, recovery_rate,
        )
        return hazard_credit_triangle(spread, recovery_rate)

    try:
        lam = brentq(
            _cds_pv_residual, _LAMBDA_LO, _LAMBDA_HI,
            args=args, xtol=1e-10, rtol=1e-10, maxiter=200,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("Brent failed for spread=%.6f, R=%.2f: %s",
                       spread, recovery_rate, exc)
        return np.nan

    return float(lam)


# ---------------------------------------------------------------------------
# Vectorised pipeline wrapper
# ---------------------------------------------------------------------------

_REQUIRED_COLS = ["parspread"]


def run_hull_white(
    cds_df: pd.DataFrame,
    recovery_rate: Optional[float] = None,
    tenor: Optional[float] = None,
    horizon: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    payments_per_year: Optional[int] = None,
) -> pd.DataFrame:
    """Apply the CDS-bootstrap Hull-White model to a panel of CDS quotes.

    Recovery rate selection (per row, in priority order):
        1. ``estimated_rr`` column  — firm-specific RR from balance sheet
           (total_assets / total_debt, clipped to [0.05, 0.95]).
        2. ``recovery_rate`` argument — scalar fallback.
        3. ``config.RECOVERY_RATE``   — global default (0.40).

    Parameters
    ----------
    cds_df : pd.DataFrame
        CDS panel. Must contain ``parspread``.
        Optional: ``estimated_rr``, ``dp``, ``ticker``, ``year``.
    recovery_rate : float, optional
        Scalar fallback RR when ``estimated_rr`` is missing.
    tenor, horizon, risk_free_rate, payments_per_year
        Standard CDS pricing parameters.

    Returns
    -------
    pd.DataFrame
        Input augmented with:
        - hw_lambda_triangle  : credit-triangle hazard
        - hw_lambda           : PV-bootstrapped hazard (preferred)
        - hw_PD_1y            : 1-year PD
        - hw_PD_5y            : 5-year PD
        - hw_PD_triangle_5y   : 5-year PD from triangle hazard
        - hw_recovery_rate    : actual RR used per row
        - hw_dp_residual      : hw_PD_5y - dp  (if dp present)
    """
    df = cds_df.copy()

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    fallback_rr = float(
        recovery_rate if recovery_rate is not None else config.RECOVERY_RATE
    )
    if not (0.0 <= fallback_rr < 1.0):
        raise ValueError(f"recovery_rate must be in [0, 1); got {fallback_rr}")

    tenor_eff   = float(tenor   if tenor   is not None else config.CDS_TENOR)
    horizon_eff = float(horizon if horizon is not None else config.T)
    ppy_eff     = int(payments_per_year if payments_per_year is not None
                      else config.CDS_PAYMENTS_PER_YEAR)

    n = len(df)
    if n == 0:
        logger.warning("Empty CDS DataFrame — returning empty result.")
        for c in ["hw_lambda_triangle", "hw_lambda", "hw_PD_1y",
                  "hw_PD_5y", "hw_PD_triangle_5y", "hw_recovery_rate"]:
            df[c] = np.nan
        return df

    # ------------------------------------------------------------------
    # Build per-row RR array
    # ------------------------------------------------------------------
    if "estimated_rr" in df.columns:
        rr_array = (
            df["estimated_rr"]
            .fillna(fallback_rr)
            .clip(lower=config.RR_LO, upper=config.RR_HI)
            .values.astype(float)
        )
        n_firm = int(df["estimated_rr"].notna().sum())
        n_fall = n - n_firm
        logger.info(
            "RR: %d firm-level (estimated_rr), %d fallback (%.2f)",
            n_firm, n_fall, fallback_rr,
        )
    else:
        rr_array = np.full(n, fallback_rr, dtype=float)
        logger.info(
            "RR: no estimated_rr column — using global fallback %.2f "
            "for all %d rows.", fallback_rr, n,
        )

    # ------------------------------------------------------------------
    # Build per-row risk-free rate array
    #   priority: 1) panel column 'risk_free_rate', 2) scalar argument
    # ------------------------------------------------------------------
    if "risk_free_rate" in df.columns:
        scalar_r = float(risk_free_rate) if risk_free_rate is not None else np.nan
        r_array = (
            df["risk_free_rate"]
            .fillna(scalar_r if np.isfinite(scalar_r) else 0.0)
            .values.astype(float)
        )
        logger.info("Risk-free: per-row from panel column 'risk_free_rate'")
    elif risk_free_rate is not None:
        r_array = np.full(n, float(risk_free_rate), dtype=float)
        logger.info("Risk-free: scalar argument %.4f for all rows", risk_free_rate)
    else:
        raise ValueError(
            "No risk-free rate available: pass `risk_free_rate=...` "
            "or include a `risk_free_rate` column in cds_df."
        )

    # ------------------------------------------------------------------
    # Compute hazards row-by-row using firm-level RR and r
    # ------------------------------------------------------------------
    spreads = df["parspread"].values

    lam_tri = np.array([
        hazard_credit_triangle(s, rr)
        for s, rr in zip(spreads, rr_array)
    ], dtype=float)

    lam_pv = np.array([
        hazard_pv_solve(s, rr,
                        tenor=tenor_eff,
                        risk_free_rate=r,
                        payments_per_year=ppy_eff)
        for s, rr, r in zip(spreads, rr_array, r_array)
    ], dtype=float)

    # ------------------------------------------------------------------
    # Convert to PDs
    # ------------------------------------------------------------------
    df["hw_lambda_triangle"] = lam_tri
    df["hw_lambda"]          = lam_pv
    df["hw_PD_1y"]           = 1.0 - np.exp(-lam_pv  * horizon_eff)
    df["hw_PD_5y"]           = 1.0 - np.exp(-lam_pv  * tenor_eff)
    df["hw_PD_triangle_5y"]  = 1.0 - np.exp(-lam_tri * tenor_eff)
    df["hw_recovery_rate"]   = rr_array

    # ------------------------------------------------------------------
    # Sanity check vs Markit dp
    # ------------------------------------------------------------------
    if "dp" in df.columns:
        df["hw_dp_residual"] = df["hw_PD_5y"] - df["dp"]
        res = df["hw_dp_residual"].dropna()
        if len(res):
            logger.info(
                "Sanity check: mean(hw_PD_5y - dp)=%.6f, std=%.6f, n=%d",
                res.mean(), res.std(), len(res),
            )

    n_ok = int(np.isfinite(lam_pv).sum())
    logger.info(
        "Hull-White bootstrap complete: %d/%d solved, "
        "mean RR=%.3f, tenor=%.1fy, mean r=%.4f, mean lambda=%.6f",
        n_ok, n,
        float(np.nanmean(rr_array)),
        tenor_eff,
        float(np.nanmean(r_array)),
        np.nanmean(lam_pv) if n_ok > 0 else np.nan,
    )

    return df
