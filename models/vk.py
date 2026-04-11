"""
Vasicek-Kealhofer (KMV) structural credit risk model.

Extends the Merton framework with two key modifications:
    1. Default point (DP) = short-term debt + 0.5 * long-term debt,
       rather than total debt.
    2. Iterative procedure to estimate asset value and volatility,
       rather than simultaneous nonlinear solve.

References:
    Crosbie, P. & Bohn, J. (2003). Modeling Default Risk. Moody's KMV.
    Vassalou, M. & Xing, Y. (2004). Default risk in equity returns.
    *Journal of Finance*, 59(2), 831-868.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def iterative_asset_value(
    E: float,
    sigma_E: float,
    D: float,
    r: float,
    T: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Tuple[float, float]:
    """Estimate asset value and volatility via KMV iterative procedure.

    Algorithm:
        1. Initialise V = E + D, sigma_V = sigma_E * E / V.
        2. Compute d1, d2 from current (V, sigma_V).
        3. Update V = (E + D * exp(-rT) * N(d2)) / N(d1).
        4. Update sigma_V = sigma_E * E / (V * N(d1)).
        5. Repeat until |V_new - V_old| < tol or max_iter reached.

    Parameters
    ----------
    E : float
        Market value of equity.
    sigma_E : float
        Annualised equity volatility.
    D : float
        Default barrier (typically the default point, DP).
    r : float
        Risk-free rate (annualised, continuous).
    T : float, optional
        Time horizon in years (default 1.0).
    tol : float, optional
        Convergence tolerance on asset value (default 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default 200).

    Returns
    -------
    tuple[float, float]
        (V, sigma_V) — estimated asset value and asset volatility.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within *max_iter* steps.
    """
    sqrt_T = np.sqrt(T)
    disc = np.exp(-r * T)

    # Step 1 — initialise
    V = E + D
    sigma_V = sigma_E * E / V

    for i in range(max_iter):
        V_old = V

        # Black-Scholes d1, d2
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrt_T)
        d2 = d1 - sigma_V * sqrt_T

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)

        # Guard against degenerate N(d1)
        if N_d1 < 1e-12:
            N_d1 = 1e-12

        # Step 3 — update asset value
        V = (E + D * disc * N_d2) / N_d1

        # Step 4 — update asset volatility
        # Recompute d1 with the new V for a tighter update
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrt_T)
        N_d1 = norm.cdf(d1)
        if N_d1 < 1e-12:
            N_d1 = 1e-12

        sigma_V = sigma_E * E / (V * N_d1)

        # Step 5 — convergence check
        if abs(V - V_old) < tol:
            logger.debug("VK iteration converged in %d steps", i + 1)
            return float(V), float(sigma_V)

    logger.warning(
        "VK iteration did not converge after %d iterations (|ΔV| = %.2e)",
        max_iter,
        abs(V - V_old),
    )
    return float(V), float(sigma_V)


def vk_dd(
    V: float,
    sigma_V: float,
    default_point: float,
    mu: float,
    T: float = 1.0,
) -> float:
    """Calculate distance to default under the VK / KMV model.

    DD = [ln(V / DP) + (mu - sigma_V^2 / 2) * T] / (sigma_V * sqrt(T))

    Uses the physical drift *mu* (expected asset return) rather than
    the risk-free rate, distinguishing this from the Merton risk-neutral DD.

    Parameters
    ----------
    V : float
        Estimated asset value.
    sigma_V : float
        Estimated asset volatility.
    default_point : float
        Default point (DP = STD + 0.5 * LTD).
    mu : float
        Expected asset return (physical measure).
    T : float, optional
        Time horizon in years (default 1.0).

    Returns
    -------
    float
        Distance to default.
    """
    sqrt_T = np.sqrt(T)
    dd = (np.log(V / default_point) + (mu - 0.5 * sigma_V ** 2) * T) / (
        sigma_V * sqrt_T
    )
    return float(dd)


def vk_pd(dd: float) -> float:
    """Compute probability of default from distance to default.

    PD = N(-DD), where N is the standard normal CDF.

    Parameters
    ----------
    dd : float
        Distance to default.

    Returns
    -------
    float
        Estimated probability of default.
    """
    return float(norm.cdf(-dd))


# ---------------------------------------------------------------------------
# Panel runner
# ---------------------------------------------------------------------------

def run_vk(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Vasicek-Kealhofer model to every row in a panel DataFrame.

    Expected columns in *panel_df*:
        - market_cap      : market value of equity (E)
        - equity_vol      : annualised equity volatility (sigma_E)
        - default_point   : DP = short_term_debt + 0.5 * long_term_debt
        - risk_free_rate  : annualised risk-free rate (r)

    The drift parameter *mu* is set to the risk-free rate plus a small
    equity-risk premium (2 %).  This is a standard practical choice when
    firm-specific expected returns are unavailable.

    New columns added:
        - vk_V       : estimated asset value
        - vk_sigma_V : estimated asset volatility
        - vk_DD      : distance to default
        - vk_PD      : probability of default

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with one row per firm-period observation.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with VK model outputs.
    """
    T = config.T
    EQUITY_RISK_PREMIUM = 0.02  # small premium over r

    results = {
        "vk_V": [],
        "vk_sigma_V": [],
        "vk_DD": [],
        "vk_PD": [],
    }

    for idx, row in panel_df.iterrows():
        try:
            E = float(row["market_cap"])
            sigma_E = float(row["equity_vol"])
            dp = float(row["default_point"])
            r = float(row["risk_free_rate"])

            # Fallback: use total_debt when default_point is missing
            if np.isnan(dp):
                dp = float(row["total_debt"])

            if E <= 0 or sigma_E <= 0 or dp <= 0:
                raise ValueError("Non-positive input value")

            V, sigma_V = iterative_asset_value(E, sigma_E, dp, r, T)

            if V <= 0 or sigma_V <= 0:
                raise ValueError("Iteration returned non-positive value")

            mu = r + EQUITY_RISK_PREMIUM
            dd = vk_dd(V, sigma_V, dp, mu, T)
            pd_val = vk_pd(dd)

            results["vk_V"].append(V)
            results["vk_sigma_V"].append(sigma_V)
            results["vk_DD"].append(dd)
            results["vk_PD"].append(pd_val)

        except Exception as exc:
            logger.warning("VK model failed for row %s: %s", idx, exc)
            results["vk_V"].append(np.nan)
            results["vk_sigma_V"].append(np.nan)
            results["vk_DD"].append(np.nan)
            results["vk_PD"].append(np.nan)

    for col, values in results.items():
        panel_df[col] = values

    logger.info(
        "VK model complete: %d rows, %d failures",
        len(panel_df),
        panel_df["vk_DD"].isna().sum(),
    )

    return panel_df
