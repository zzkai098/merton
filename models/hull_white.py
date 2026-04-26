"""
Hull-White (2000) reduced-form credit risk model — CDS bootstrap implementation.

This module implements the original Hull & White (2000) approach: a default
intensity (hazard rate) is *bootstrapped* from observed CDS par spreads under
the standard reduced-form pricing equation.  No accounting features are used.

Two estimators are provided for a flat (piecewise-constant, single-tenor)
hazard rate:

1. **Credit-triangle approximation** (closed form):

       lambda ≈ s / (1 - R)

   This is the textbook first-order approximation, exact in the limit of
   continuous premium payments, zero discounting and small spreads.  It is
   useful as a sanity check and for very fast computation.

2. **PV-equation numerical solver** (recommended):

   The flat hazard ``lambda`` is found by setting the protection leg PV
   equal to the premium leg PV (with accrual on default) on a quarterly
   payment grid::

       PV_protection(lambda) = PV_premium(lambda) * s

   This is solved via Brent's method.  It accounts for discounting and
   accrual-on-default and so handles wider spreads correctly.

Because the input dataset has a single tenor (5Y), the hazard is assumed
*flat* (constant in time).  A multi-tenor extension would solve for a
piecewise-constant hazard curve by bootstrapping each tenor sequentially;
that path is left as a future extension.

Reference
---------
Hull, J. & White, A. (2000). Valuing credit default swaps I: No counterparty
default risk. *Journal of Derivatives*, 8(1), 29–40.

Functions
---------
hazard_credit_triangle(spread, recovery_rate)
    Closed-form approximation lambda = s / (1 - R).
hazard_pv_solve(spread, recovery_rate, tenor, ...)
    Numerical bootstrap of flat hazard from the CDS PV equation.
run_hull_white(cds_df, recovery_rate=None, ...)
    Pipeline wrapper that operates on a DataFrame of CDS quotes.
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
# Module-level defaults
# ---------------------------------------------------------------------------

# Default CDS pricing conventions (ISDA-style).
_DEFAULT_TENOR = 5.0           # CDS tenor in years
_DEFAULT_HORIZON = 1.0         # Horizon for the 1Y PD output
_DEFAULT_PAYMENTS_PER_YEAR = 4 # Quarterly premium payments (standard)
_DEFAULT_RISK_FREE = 0.04      # Flat discount rate fallback if not provided

# Search bounds for the numerical hazard solver.  A flat hazard above 5.0
# (i.e. 500% per year) is economically meaningless and would correspond to
# spreads far outside any realistic CDS quote.
_LAMBDA_LO = 1e-8
_LAMBDA_HI = 5.0


# ---------------------------------------------------------------------------
# Estimator 1: closed-form credit-triangle approximation
# ---------------------------------------------------------------------------

def hazard_credit_triangle(
    spread: float,
    recovery_rate: float,
) -> float:
    """Closed-form credit-triangle approximation of the flat hazard rate.

    Under the standard reduced-form CDS pricing identity, ignoring
    discounting and accrual-on-default and assuming continuous payments::

        lambda ≈ s / (1 - R)

    This is exact in the limit s -> 0 and gives a useful first-order
    benchmark for the more accurate :func:`hazard_pv_solve`.

    Parameters
    ----------
    spread : float
        CDS par spread in decimal annual units (e.g. 0.005 for 50 bps).
    recovery_rate : float
        Assumed recovery on default, in [0, 1).

    Returns
    -------
    float
        Implied flat hazard rate.  Returns NaN if inputs are invalid.
    """
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
    """PV(protection leg) - spread * PV(premium leg + accrual).

    The premium leg pays ``spread * dt`` at each payment date conditional
    on no prior default.  On default between payment dates we assume the
    seller pays the accrued premium for the partial period (the standard
    accrual-on-default convention).  The protection leg pays ``(1 - R)``
    at the time of default.  All cash flows are discounted at the
    risk-free rate ``r``.

    A piecewise-constant protection-leg approximation is used: default
    within the i-th period is assumed to occur at the period midpoint.
    """
    dt = 1.0 / payments_per_year
    n = int(round(tenor * payments_per_year))

    # Payment dates t_i = i * dt for i = 1..n, midpoints m_i = (i-0.5) * dt.
    i = np.arange(1, n + 1)
    t_i = i * dt
    m_i = (i - 0.5) * dt

    # Discount factors at payment dates and midpoints.
    df_t = np.exp(-risk_free_rate * t_i)
    df_m = np.exp(-risk_free_rate * m_i)

    # Survival probabilities under flat hazard.
    surv_t = np.exp(-lam * t_i)
    surv_t_prev = np.exp(-lam * (t_i - dt))   # survival at t_{i-1}
    p_default_in_i = surv_t_prev - surv_t     # default within period i

    # Premium leg: pay spread*dt at t_i if survived, plus accrual of
    # (dt/2)*spread if default occurred mid-period.
    pv_premium_regular = np.sum(df_t * surv_t * dt)
    pv_premium_accrual = np.sum(df_m * p_default_in_i * (dt / 2.0))
    pv_premium = pv_premium_regular + pv_premium_accrual

    # Protection leg: pay (1-R) at midpoint of the period in which default
    # occurs.
    pv_protection = (1.0 - recovery_rate) * np.sum(df_m * p_default_in_i)

    return pv_protection - spread * pv_premium


def hazard_pv_solve(
    spread: float,
    recovery_rate: float,
    tenor: float = _DEFAULT_TENOR,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
    payments_per_year: int = _DEFAULT_PAYMENTS_PER_YEAR,
) -> float:
    """Bootstrap a flat hazard rate from a single CDS par spread.

    Solves for ``lambda`` such that the protection leg PV equals the
    premium leg PV (including accrual-on-default) using Brent's method.

    Parameters
    ----------
    spread : float
        CDS par spread in decimal annual units.
    recovery_rate : float
        Assumed recovery on default, in [0, 1).
    tenor : float, optional
        CDS tenor in years (default 5.0).
    risk_free_rate : float, optional
        Flat continuously-compounded discount rate (default 0.04).
    payments_per_year : int, optional
        Premium payment frequency (default 4 = quarterly).

    Returns
    -------
    float
        Implied flat hazard rate, or NaN if inputs are invalid or the
        solver fails to bracket a root.
    """
    # Input validation
    if not np.isfinite(spread) or not np.isfinite(recovery_rate):
        return np.nan
    if spread <= 0 or recovery_rate < 0 or recovery_rate >= 1.0:
        return np.nan
    if tenor <= 0 or payments_per_year < 1:
        return np.nan

    args = (spread, recovery_rate, tenor, risk_free_rate, payments_per_year)

    # Verify the residual changes sign over [_LAMBDA_LO, _LAMBDA_HI].
    f_lo = _cds_pv_residual(_LAMBDA_LO, *args)
    f_hi = _cds_pv_residual(_LAMBDA_HI, *args)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return np.nan
    if f_lo * f_hi > 0:
        # No sign change — return triangle approximation as fallback.
        logger.warning(
            "PV residual does not change sign for spread=%.6f, R=%.2f; "
            "falling back to credit-triangle approximation.",
            spread, recovery_rate,
        )
        return hazard_credit_triangle(spread, recovery_rate)

    try:
        lam = brentq(
            _cds_pv_residual,
            _LAMBDA_LO,
            _LAMBDA_HI,
            args=args,
            xtol=1e-10,
            rtol=1e-10,
            maxiter=200,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "Brent solver failed for spread=%.6f, R=%.2f: %s",
            spread, recovery_rate, exc,
        )
        return np.nan

    return float(lam)


# ---------------------------------------------------------------------------
# Vectorised pipeline wrapper
# ---------------------------------------------------------------------------

# Required input columns
_REQUIRED_COLS = ["parspread"]
# Optional columns used when present
_OPTIONAL_COLS = ["dp", "year", "ticker"]


def run_hull_white(
    cds_df: pd.DataFrame,
    recovery_rate: Optional[float] = None,
    tenor: float = _DEFAULT_TENOR,
    horizon: float = _DEFAULT_HORIZON,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
    payments_per_year: int = _DEFAULT_PAYMENTS_PER_YEAR,
) -> pd.DataFrame:
    """Apply the CDS-bootstrap Hull-White model to a panel of CDS quotes.

    For each row in ``cds_df`` (one per ticker-year), the function:

    1. Computes the credit-triangle hazard ``lambda ≈ s / (1 - R)``.
    2. Bootstraps a more accurate hazard from the CDS PV equation via
       :func:`hazard_pv_solve`.
    3. Converts both hazards into 1-year and 5-year default probabilities
       under the standard survival framework ``PD(T) = 1 - exp(-lambda * T)``.
    4. If a ``dp`` column is present (Markit-implied 5Y PD), also reports
       the residual between ``hw_PD_5y`` and ``dp`` as a sanity check.

    Parameters
    ----------
    cds_df : pd.DataFrame
        CDS panel.  Must contain a ``parspread`` column (decimal annual
        spread).  May optionally contain ``ticker``, ``year``, ``dp``.
    recovery_rate : float, optional
        Recovery rate.  If None (default), uses ``config.RECOVERY_RATE``.
    tenor : float, optional
        CDS tenor in years (default 5.0).
    horizon : float, optional
        Horizon T for the 1Y PD output (default 1.0).
    risk_free_rate : float, optional
        Discount rate for the PV solver (default 0.04).
    payments_per_year : int, optional
        Premium payment frequency (default 4 = quarterly).

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with the following columns:

        - ``hw_lambda_triangle`` : credit-triangle hazard
        - ``hw_lambda``          : PV-bootstrapped hazard (preferred)
        - ``hw_PD_1y``           : 1-year PD from ``hw_lambda``
        - ``hw_PD_5y``           : 5-year PD from ``hw_lambda``
        - ``hw_PD_triangle_5y``  : 5-year PD from triangle hazard
        - ``hw_recovery_rate``   : recovery rate used (for traceability)
        - ``hw_dp_residual``     : ``hw_PD_5y - dp`` (only if ``dp`` present)
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    df = cds_df.copy()

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    if recovery_rate is None:
        recovery_rate = float(config.RECOVERY_RATE)

    if not (0.0 <= recovery_rate < 1.0):
        raise ValueError(
            f"recovery_rate must be in [0, 1); got {recovery_rate}"
        )

    n = len(df)
    if n == 0:
        logger.warning("Empty CDS DataFrame — returning empty result.")
        for c in [
            "hw_lambda_triangle", "hw_lambda",
            "hw_PD_1y", "hw_PD_5y", "hw_PD_triangle_5y",
            "hw_recovery_rate",
        ]:
            df[c] = np.nan
        return df

    # ------------------------------------------------------------------
    # Compute hazards row-by-row
    # ------------------------------------------------------------------
    spreads = df["parspread"].values
    lam_tri = np.array([
        hazard_credit_triangle(s, recovery_rate) for s in spreads
    ], dtype=float)

    lam_pv = np.array([
        hazard_pv_solve(
            s,
            recovery_rate,
            tenor=tenor,
            risk_free_rate=risk_free_rate,
            payments_per_year=payments_per_year,
        )
        for s in spreads
    ], dtype=float)

    # ------------------------------------------------------------------
    # Convert hazards to PDs
    # ------------------------------------------------------------------
    pd_1y = 1.0 - np.exp(-lam_pv * horizon)
    pd_5y = 1.0 - np.exp(-lam_pv * tenor)
    pd_tri_5y = 1.0 - np.exp(-lam_tri * tenor)

    df["hw_lambda_triangle"] = lam_tri
    df["hw_lambda"] = lam_pv
    df["hw_PD_1y"] = pd_1y
    df["hw_PD_5y"] = pd_5y
    df["hw_PD_triangle_5y"] = pd_tri_5y
    df["hw_recovery_rate"] = recovery_rate

    # ------------------------------------------------------------------
    # Sanity check vs Markit dp (if available)
    # ------------------------------------------------------------------
    if "dp" in df.columns:
        df["hw_dp_residual"] = df["hw_PD_5y"] - df["dp"]
        residual_finite = df["hw_dp_residual"].dropna()
        if len(residual_finite) > 0:
            logger.info(
                "Sanity check: mean(hw_PD_5y - dp) = %.6f, "
                "std = %.6f over %d rows",
                residual_finite.mean(),
                residual_finite.std(),
                len(residual_finite),
            )

    n_ok = int(np.isfinite(lam_pv).sum())
    logger.info(
        "Hull-White CDS bootstrap complete: %d/%d rows solved, "
        "R=%.2f, tenor=%.1fy, mean lambda=%.6f",
        n_ok, n, recovery_rate, tenor,
        np.nanmean(lam_pv) if n_ok > 0 else np.nan,
    )

    return df
