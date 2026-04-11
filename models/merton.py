"""
Merton (1974) structural credit risk model.

Models equity as a European call option on firm assets. Solves for
unobservable asset value (V) and asset volatility (sigma_V) from
observable equity value and equity volatility using two simultaneous
nonlinear equations.

References:
    Merton, R. C. (1974). On the pricing of corporate debt: The risk
    structure of interest rates. *Journal of Finance*, 29(2), 449-470.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def solve_merton(
    E: float,
    sigma_E: float,
    D: float,
    r: float,
    T: float = 1.0,
) -> Tuple[float, float]:
    """Solve for asset value and asset volatility via fsolve.

    The two nonlinear equations are:
        1) E = V * N(d1) - D * exp(-rT) * N(d2)          (call option)
        2) sigma_E * E = V * N(d1) * sigma_V              (volatility relation)

    Parameters
    ----------
    E : float
        Market value of equity (market cap).
    sigma_E : float
        Annualised equity volatility.
    D : float
        Face value of debt (default barrier).
    r : float
        Risk-free interest rate (annualised, continuous).
    T : float, optional
        Time to maturity in years (default 1.0).

    Returns
    -------
    tuple[float, float]
        (V, sigma_V) — estimated asset value and asset volatility.
    """

    def _equations(x):
        V, sigma_V = x
        if V <= 0 or sigma_V <= 0:
            return [1e6, 1e6]

        sqrt_T = np.sqrt(T)
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrt_T)
        d2 = d1 - sigma_V * sqrt_T

        eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
        eq2 = V * norm.cdf(d1) * sigma_V - sigma_E * E

        return [eq1, eq2]

    # Initial guesses
    V0 = E + D
    sigma_V0 = sigma_E * E / (E + D)

    solution, info, ier, msg = fsolve(_equations, [V0, sigma_V0], full_output=True)

    if ier != 1:
        logger.warning("Merton solver did not converge: %s", msg)

    V_sol, sigma_V_sol = solution
    return float(V_sol), float(sigma_V_sol)


def merton_dd(
    V: float,
    sigma_V: float,
    D: float,
    r: float,
    T: float = 1.0,
) -> float:
    """Calculate distance to default under the Merton model.

    DD = [ln(V/D) + (r - sigma_V^2 / 2) * T] / (sigma_V * sqrt(T))

    This equals d2 under the risk-neutral measure.

    Parameters
    ----------
    V : float
        Estimated asset value.
    sigma_V : float
        Estimated asset volatility.
    D : float
        Face value of debt (default barrier).
    r : float
        Risk-free rate (annualised, continuous).
    T : float, optional
        Time horizon in years (default 1.0).

    Returns
    -------
    float
        Distance to default.
    """
    sqrt_T = np.sqrt(T)
    dd = (np.log(V / D) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrt_T)
    return float(dd)


def merton_pd(dd: float) -> float:
    """Compute probability of default from distance to default.

    PD = N(-DD), where N is the standard normal CDF.

    Parameters
    ----------
    dd : float
        Distance to default.

    Returns
    -------
    float
        Risk-neutral probability of default.
    """
    return float(norm.cdf(-dd))


# ---------------------------------------------------------------------------
# Panel runner
# ---------------------------------------------------------------------------

def run_merton(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Merton model to every row in a panel DataFrame.

    Expected columns in *panel_df*:
        - market_cap     : market value of equity (E)
        - equity_vol     : annualised equity volatility (sigma_E)
        - total_debt     : face value of total debt (D)
        - risk_free_rate : annualised risk-free rate (r)

    New columns added:
        - merton_V       : estimated asset value
        - merton_sigma_V : estimated asset volatility
        - merton_DD      : distance to default
        - merton_PD      : probability of default

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with one row per firm-period observation.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with Merton model outputs.
    """
    T = config.T

    results = {
        "merton_V": [],
        "merton_sigma_V": [],
        "merton_DD": [],
        "merton_PD": [],
    }

    for idx, row in panel_df.iterrows():
        try:
            E = float(row["market_cap"])
            sigma_E = float(row["equity_vol"])
            D = float(row["total_debt"])
            r = float(row["risk_free_rate"])

            if E <= 0 or sigma_E <= 0 or D <= 0:
                raise ValueError("Non-positive input value")

            V, sigma_V = solve_merton(E, sigma_E, D, r, T)

            if V <= 0 or sigma_V <= 0:
                raise ValueError("Solver returned non-positive value")

            dd = merton_dd(V, sigma_V, D, r, T)
            pd_val = merton_pd(dd)

            results["merton_V"].append(V)
            results["merton_sigma_V"].append(sigma_V)
            results["merton_DD"].append(dd)
            results["merton_PD"].append(pd_val)

        except Exception as exc:
            logger.warning("Merton model failed for row %s: %s", idx, exc)
            results["merton_V"].append(np.nan)
            results["merton_sigma_V"].append(np.nan)
            results["merton_DD"].append(np.nan)
            results["merton_PD"].append(np.nan)

    for col, values in results.items():
        panel_df[col] = values

    logger.info(
        "Merton model complete: %d rows, %d failures",
        len(panel_df),
        panel_df["merton_DD"].isna().sum(),
    )

    return panel_df
