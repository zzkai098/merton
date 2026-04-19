"""
Merton (1974) structural credit risk model.

Models equity as a European call option on firm assets. Solves for
unobservable asset value (V) and asset volatility (sigma_V) from
observable equity value and equity volatility using two simultaneous
nonlinear equations.

Key improvement over v1
-----------------------
sigma_E input now uses implied volatility (from OptionMetrics) where
available, falling back to 252-day historical volatility otherwise.
The source of sigma_E is recorded in the ``merton_vol_source`` column
for transparency and robustness analysis.

References:
    Merton, R. C. (1974). On the pricing of corporate debt: The risk
    structure of interest rates. *Journal of Finance*, 29(2), 449-470.
    Bharath, S. T. & Shumway, T. (2008). Forecasting default with the
    Merton distance to default model. *Review of Financial Studies*,
    21(3), 1339-1369.
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
# Sanity-check bounds
# ---------------------------------------------------------------------------
_SIGMA_E_MAX = 5.0   # 500% annualised equity vol — anything above is suspect
_SIGMA_V_MAX = 5.0   # same cap on the solved asset vol


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

    Multiple initial guesses are tried in sequence; the first converged
    solution is returned. This improves robustness for distressed firms
    where E is very small relative to D.

    Parameters
    ----------
    E : float
        Market value of equity (market cap).
    sigma_E : float
        Annualised equity volatility (implied or historical).
    D : float
        Face value of debt (default barrier).
    r : float
        Risk-free interest rate (annualised, continuous).
    T : float, optional
        Time to maturity in years (default 1.0). Panel data uses T = 1.

    Returns
    -------
    tuple[float, float]
        (V, sigma_V) — estimated asset value and asset volatility.
        Returns (nan, nan) if all guesses fail to converge.
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

    # Multiple initial guesses — ordered from most to least informative.
    # The fourth guess handles severely distressed firms where E ≈ 0.
    initial_guesses = [
        (E + D,           sigma_E * E / (E + D)),   # classic Merton initialisation
        (E + D,           sigma_E),                  # let sigma start at equity vol
        (max(E, D) * 1.5, sigma_E * 0.5),            # wider asset base, lower vol
        (D,               sigma_E),                  # distressed: V near barrier
    ]

    for V0, sigma_V0 in initial_guesses:
        try:
            solution, _, ier, msg = fsolve(
                _equations, [V0, sigma_V0], full_output=True
            )
            if ier == 1:
                V_sol, sigma_V_sol = solution
                if V_sol > 0 and 0 < sigma_V_sol <= _SIGMA_V_MAX:
                    return float(V_sol), float(sigma_V_sol)
        except Exception:
            continue

    logger.warning(
        "Merton solver did not converge for E=%.4g, sigma_E=%.4g, D=%.4g",
        E, sigma_E, D,
    )
    return float("nan"), float("nan")


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
        Distance to default (= d2, the risk-neutral default threshold).
    """
    sqrt_T = np.sqrt(T)
    dd = (np.log(V / D) + (r - 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrt_T)
    return float(dd)


def merton_pd(dd: float) -> float:
    """Compute risk-neutral probability of default from distance to default.

    PD = N(-DD), where N is the standard normal CDF.

    Parameters
    ----------
    dd : float
        Distance to default.

    Returns
    -------
    float
        Risk-neutral probability of default in [0, 1].
    """
    return float(norm.cdf(-dd))


# ---------------------------------------------------------------------------
# Panel runner
# ---------------------------------------------------------------------------

def run_merton(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Merton model to every row in a panel DataFrame.

    sigma_E selection (in priority order):
        1. impl_volatility  — OptionMetrics ATM implied vol (forward-looking,
                              delta ≈ 0.5, ~1-year maturity). Preferred because
                              it reflects the market's current expectation of
                              future volatility, consistent with the option-
                              pricing foundation of the Merton framework
                              (Bharath & Shumway, 2008).
        2. equity_vol       — 252-day historical volatility. Used as fallback
                              when implied vol is unavailable (e.g. distressed
                              firms that ceased options trading in 2025).

    Expected columns in *panel_df*:
        - market_cap       : market value of equity (E)
        - equity_vol       : annualised historical equity volatility
        - impl_volatility  : ATM implied volatility (may be NaN)
        - total_debt       : face value of total debt (D)
        - risk_free_rate   : annualised risk-free rate (r)

    New columns added:
        - merton_V          : estimated asset value
        - merton_sigma_V    : estimated asset volatility
        - merton_sigma_E    : sigma_E actually used (implied or historical)
        - merton_vol_source : 'implied' or 'historical'
        - merton_DD         : distance to default (= d2, risk-neutral)
        - merton_PD         : risk-neutral probability of default

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with one row per firm-year observation.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with Merton model outputs.
    """
    T = config.T  # 1.0 for annual panel; extend to 0.25 for quarterly

    results: dict[str, list] = {
        "merton_V":          [],
        "merton_sigma_V":    [],
        "merton_sigma_E":    [],
        "merton_vol_source": [],
        "merton_DD":         [],
        "merton_PD":         [],
    }

    for idx, row in panel_df.iterrows():
        try:
            E = float(row["market_cap"])
            D = float(row["total_debt"])
            r = float(row["risk_free_rate"])

            # --- sigma_E selection -----------------------------------------
            # Priority 1: implied volatility (forward-looking, preferred)
            # Priority 2: historical volatility (fallback)
            impl_vol = row.get("impl_volatility")
            hist_vol = row.get("equity_vol")

            if impl_vol is not None and not (
                isinstance(impl_vol, float) and np.isnan(impl_vol)
            ):
                sigma_E    = float(impl_vol)
                vol_source = "implied"
            else:
                sigma_E    = float(hist_vol)
                vol_source = "historical"

            # --- Input validation ------------------------------------------
            if any(np.isnan([E, sigma_E, D, r])):
                raise ValueError("NaN in input values")

            if E <= 0 or sigma_E <= 0 or D <= 0:
                raise ValueError(
                    f"Non-positive input: E={E}, sigma_E={sigma_E}, D={D}"
                )

            if sigma_E > _SIGMA_E_MAX:
                raise ValueError(
                    f"sigma_E={sigma_E:.2f} exceeds cap of {_SIGMA_E_MAX}"
                )

            # --- Solve --------------------------------------------------------
            V, sigma_V = solve_merton(E, sigma_E, D, r, T)

            if np.isnan(V) or np.isnan(sigma_V):
                raise ValueError("Solver returned NaN")

            dd     = merton_dd(V, sigma_V, D, r, T)
            pd_val = merton_pd(dd)

            results["merton_V"].append(V)
            results["merton_sigma_V"].append(sigma_V)
            results["merton_sigma_E"].append(sigma_E)
            results["merton_vol_source"].append(vol_source)
            results["merton_DD"].append(dd)
            results["merton_PD"].append(pd_val)

        except Exception as exc:
            logger.warning("Merton failed for row %s: %s", idx, exc)
            results["merton_V"].append(np.nan)
            results["merton_sigma_V"].append(np.nan)
            results["merton_sigma_E"].append(np.nan)
            results["merton_vol_source"].append("failed")
            results["merton_DD"].append(np.nan)
            results["merton_PD"].append(np.nan)

    for col, values in results.items():
        panel_df[col] = values

    n_total    = len(panel_df)
    n_failures = int(panel_df["merton_DD"].isna().sum())
    n_success  = n_total - n_failures
    n_implied  = int((panel_df["merton_vol_source"] == "implied").sum())
    n_hist     = int((panel_df["merton_vol_source"] == "historical").sum())

    logger.info(
        "Merton model complete: %d / %d rows solved (%.1f%% convergence) — "
        "%d implied vol, %d historical vol fallback, %d failures",
        n_success, n_total,
        100 * n_success / n_total if n_total > 0 else 0.0,
        n_implied, n_hist, n_failures,
    )

    return panel_df