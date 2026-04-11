"""
Model discrimination evaluation for Merton, Vasicek-Kealhofer (VK),
and Hull-White structural credit risk models.

Provides ROC-AUC, CAP-AR metrics, summary tables, and text reports
comparing how well each model separates defaulting from non-defaulting firms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aligned_arrays(
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Drop rows where *any* model has NaN and return clean arrays.

    Parameters
    ----------
    y_true : np.ndarray
        Binary default labels (1 = default, 0 = non-default).
    scores_dict : dict[str, np.ndarray]
        Mapping of model name to PD array, all same length as *y_true*.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        Cleaned label array and cleaned scores dictionary.
    """
    y = np.asarray(y_true, dtype=float)
    stacked = np.column_stack([y] + [np.asarray(v, dtype=float) for v in scores_dict.values()])
    mask = ~np.isnan(stacked).any(axis=1)
    y_clean = y[mask].astype(int)
    clean: dict[str, np.ndarray] = {}
    for i, name in enumerate(scores_dict):
        clean[name] = stacked[mask, i + 1]
    return y_clean, clean


def _cap_area(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the CAP curve via the trapezoidal rule.

    Firms are sorted by *scores* descending. The CAP curve plots the
    cumulative fraction of defaults captured (y-axis) against the cumulative
    fraction of all firms examined (x-axis).
    """
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n = len(y_sorted)
    n_defaults = y_sorted.sum()
    if n_defaults == 0 or n == 0:
        return 0.0
    cum_defaults = np.cumsum(y_sorted)
    x = np.arange(1, n + 1) / n
    y = cum_defaults / n_defaults
    # Prepend origin (0, 0) for correct area calculation
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], y])
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_roc_auc(
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute ROC AUC for each model.

    Parameters
    ----------
    y_true : np.ndarray
        Binary default labels (1 = default, 0 = non-default).
    scores_dict : dict[str, np.ndarray]
        Model name -> predicted PD array.

    Returns
    -------
    dict[str, float]
        Model name -> AUC score.
    """
    y_clean, clean = _aligned_arrays(y_true, scores_dict)
    results: dict[str, float] = {}
    for name, pds in clean.items():
        results[name] = float(roc_auc_score(y_clean, pds))
    return results


def compute_cap_ar(
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute the Accuracy Ratio (AR) from CAP curves for each model.

    AR = (A_model - A_random) / (A_perfect - A_random)

    Parameters
    ----------
    y_true : np.ndarray
        Binary default labels (1 = default, 0 = non-default).
    scores_dict : dict[str, np.ndarray]
        Model name -> predicted PD array.

    Returns
    -------
    dict[str, float]
        Model name -> AR value in [0, 1] (higher is better).
    """
    y_clean, clean = _aligned_arrays(y_true, scores_dict)
    n = len(y_clean)
    n_defaults = int(y_clean.sum())
    if n == 0 or n_defaults == 0:
        return {name: 0.0 for name in scores_dict}

    # Random model area = 0.5 (diagonal)
    a_random = 0.5

    # Perfect model: all defaults ranked first
    perfect_scores = y_clean.astype(float)  # 1s first when sorted desc
    a_perfect = _cap_area(y_clean, perfect_scores)

    results: dict[str, float] = {}
    for name, pds in clean.items():
        a_model = _cap_area(y_clean, pds)
        denom = a_perfect - a_random
        results[name] = float((a_model - a_random) / denom) if denom != 0 else 0.0
    return results


def summary_table(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table of model outputs per firm.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with columns: ticker, default_label, merton_DD, merton_PD,
        vk_DD, vk_PD, hw_PD.

    Returns
    -------
    pd.DataFrame
        Summary table sorted by merton_PD descending.
    """
    cols = ["ticker", "default_label", "merton_DD", "merton_PD", "vk_DD", "vk_PD", "hw_PD"]
    available = [c for c in cols if c in panel_df.columns]
    df = panel_df[available].copy()
    if "merton_PD" in df.columns:
        df = df.sort_values("merton_PD", ascending=False).reset_index(drop=True)
    return df


def model_comparison_report(panel_df: pd.DataFrame) -> str:
    """Generate a text report comparing Merton, VK, and Hull-White models.

    The report includes ROC-AUC, Accuracy Ratio, observation count, and
    identifies the best-performing model on each metric.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with columns: default_label, merton_PD, vk_PD, hw_PD.

    Returns
    -------
    str
        Formatted text report.
    """
    y_true = panel_df["default_label"].values

    scores: dict[str, np.ndarray] = {}
    col_map = {"Merton": "merton_PD", "VK": "vk_PD", "HW": "hw_PD"}
    for name, col in col_map.items():
        if col in panel_df.columns:
            scores[name] = panel_df[col].values

    if not scores:
        return "No PD columns found in panel_df."

    auc = compute_roc_auc(y_true, scores)
    ar = compute_cap_ar(y_true, scores)

    n_obs = len(panel_df)
    n_defaults = int(y_true.sum())
    n_non_defaults = n_obs - n_defaults

    best_auc_model = max(auc, key=auc.get)  # type: ignore[arg-type]
    best_ar_model = max(ar, key=ar.get)  # type: ignore[arg-type]

    lines = [
        "=" * 60,
        "  Model Discrimination Comparison Report",
        "=" * 60,
        "",
        f"  Observations : {n_obs}",
        f"  Defaults     : {n_defaults}",
        f"  Non-defaults : {n_non_defaults}",
        "",
        "-" * 60,
        f"  {'Model':<10} {'ROC-AUC':>10} {'Accuracy Ratio':>16}",
        "-" * 60,
    ]
    for name in scores:
        lines.append(f"  {name:<10} {auc[name]:>10.4f} {ar[name]:>16.4f}")
    lines.append("-" * 60)
    lines.append("")
    lines.append(f"  Best ROC-AUC       : {best_auc_model} ({auc[best_auc_model]:.4f})")
    lines.append(f"  Best Accuracy Ratio: {best_ar_model} ({ar[best_ar_model]:.4f})")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
