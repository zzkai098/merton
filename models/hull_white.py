"""
Reduced-form (intensity-based) credit risk model — logistic hazard approach.

In the absence of CDS spread data, this module estimates a hazard-rate proxy
using logistic regression on accounting / market features.  The predicted
probability is interpreted as an annual hazard rate lambda, and the default
probability is derived from the survival-probability framework:

    PD = 1 - exp(-lambda * T)

This is conceptually aligned with Hull-White style intensity models but
substitutes a statistical estimate of the hazard function for the
calibration-from-spreads step.

References:
    Hull, J. & White, A. (2000). Valuing credit default swaps I: No
    counterparty default risk. *Journal of Derivatives*, 8(1), 29-40.
    Shumway, T. (2001). Forecasting bankruptcy more accurately: A simple
    hazard model. *Journal of Business*, 74(1), 101-124.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

import config

logger = logging.getLogger(__name__)

# Features used by the logistic hazard model
_FEATURE_COLS = ["leverage_ratio", "equity_vol", "log_market_cap"]


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def estimate_hazard_logistic(
    panel_df: pd.DataFrame,
    T: float = 1.0,
) -> pd.DataFrame:
    """Estimate hazard rates and default probabilities via logistic regression.

    The model uses three features:
        - leverage_ratio : D / (E + D), where D = total debt, E = market cap
        - equity_vol     : annualised equity volatility
        - log_market_cap : natural log of market capitalisation

    A LogisticRegression with ``class_weight='balanced'`` is fitted to account
    for the rarity of defaults.  To avoid in-sample bias the predicted
    probabilities are obtained using 5-fold cross-validation
    (``cross_val_predict``).

    The predicted probability is interpreted as an annual hazard rate *lambda*,
    and the default probability follows from the survival framework:

        hw_PD = 1 - exp(-lambda * T)

    New columns added:
        - hw_hazard : estimated annual hazard rate (lambda)
        - hw_PD     : estimated probability of default over horizon T

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data.  Must contain columns ``leverage_ratio``, ``equity_vol``,
        ``log_market_cap``, and ``default_label`` (binary 0/1).
    T : float, optional
        Time horizon in years (default 1.0).

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with ``hw_hazard`` and ``hw_PD``.
    """
    df = panel_df.copy()

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------
    required = _FEATURE_COLS + ["default_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------------
    # Prepare feature matrix and target
    # ------------------------------------------------------------------
    mask = df[_FEATURE_COLS + ["default_label"]].notna().all(axis=1)
    valid_idx = df.index[mask]
    n_valid = len(valid_idx)

    if n_valid == 0:
        logger.warning("No valid rows for logistic hazard model")
        df["hw_hazard"] = np.nan
        df["hw_PD"] = np.nan
        return df

    X = df.loc[valid_idx, _FEATURE_COLS].values.astype(np.float64)
    y = df.loc[valid_idx, "default_label"].values.astype(int)

    n_defaults = y.sum()
    n_non_defaults = len(y) - n_defaults

    if n_defaults == 0 or n_non_defaults == 0:
        logger.warning(
            "Only one class present (defaults=%d, non-defaults=%d); "
            "cannot fit logistic model — setting hazard to 0",
            n_defaults,
            n_non_defaults,
        )
        df["hw_hazard"] = 0.0
        df["hw_PD"] = 0.0
        return df

    # ------------------------------------------------------------------
    # Fit logistic regression with cross-validated predictions
    # ------------------------------------------------------------------
    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    # Determine appropriate number of CV folds
    min_class_count = min(n_defaults, n_non_defaults)
    cv_folds = min(5, min_class_count)
    if cv_folds < 2:
        # Fallback: fit on the full sample (small dataset edge case)
        logger.warning(
            "Too few samples for CV (min class count=%d); "
            "fitting on full sample",
            min_class_count,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]
    else:
        proba = cross_val_predict(
            clf,
            X,
            y,
            cv=cv_folds,
            method="predict_proba",
        )[:, 1]

    # Interpret predicted probability as annual hazard rate
    hazard = proba.astype(np.float64)
    hw_pd = 1.0 - np.exp(-hazard * T)

    # ------------------------------------------------------------------
    # Write results back
    # ------------------------------------------------------------------
    df["hw_hazard"] = np.nan
    df["hw_PD"] = np.nan
    df.loc[valid_idx, "hw_hazard"] = hazard
    df.loc[valid_idx, "hw_PD"] = hw_pd

    logger.info(
        "Hull-White logistic hazard model complete: %d valid rows, "
        "mean hazard=%.4f, mean PD=%.4f",
        n_valid,
        np.nanmean(hazard),
        np.nanmean(hw_pd),
    )

    return df


# ---------------------------------------------------------------------------
# Panel runner
# ---------------------------------------------------------------------------

def run_hull_white(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the reduced-form logistic hazard model to a panel DataFrame.

    This is a convenience wrapper around :func:`estimate_hazard_logistic`.
    See that function's docstring for required columns and output details.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with features and ``default_label``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with ``hw_hazard`` and ``hw_PD``.
    """
    T = config.T
    return estimate_hazard_logistic(panel_df, T=T)
