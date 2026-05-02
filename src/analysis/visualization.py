"""
Visualization module for credit risk model evaluation.

Generates ROC curves, CAP curves, distance-to-default distributions,
and PD pairwise comparison charts for Merton, VK, and Hull-White models.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import config
from analysis.discrimination import compute_roc_auc, compute_cap_ar

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"
_COLORS: dict[str, str] = {"Merton": "#2196F3", "VK": "#FF9800", "HW": "#4CAF50"}
_DPI = 150


def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they do not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _aligned_arrays(
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Drop rows where any model has NaN; return clean arrays."""
    y = np.asarray(y_true, dtype=float)
    stacked = np.column_stack([y] + [np.asarray(v, dtype=float) for v in scores_dict.values()])
    mask = ~np.isnan(stacked).any(axis=1)
    y_clean = y[mask].astype(int)
    clean: dict[str, np.ndarray] = {}
    for i, name in enumerate(scores_dict):
        clean[name] = stacked[mask, i + 1]
    return y_clean, clean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
    save_path: str = "output/roc_curves.png",
) -> None:
    """Plot ROC curves for all models on a single figure.

    Parameters
    ----------
    y_true : np.ndarray
        Binary default labels (1 = default, 0 = non-default).
    scores_dict : dict[str, np.ndarray]
        Model name -> predicted PD array.
    save_path : str
        File path for the saved figure.
    """
    plt.style.use(_STYLE)
    y_clean, clean = _aligned_arrays(y_true, scores_dict)
    auc_scores = compute_roc_auc(y_true, scores_dict)

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, pds in clean.items():
        fpr, tpr, _ = roc_curve(y_clean, pds)
        color = _COLORS.get(name, None)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {auc_scores[name]:.3f})")

    # Random reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves \u2014 Default Discrimination")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI)
    plt.close(fig)


def plot_cap_curves(
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
    save_path: str = "output/cap_curves.png",
) -> None:
    """Plot Cumulative Accuracy Profile curves for all models.

    Includes the perfect model and random baseline along with each
    model's AR in the legend.

    Parameters
    ----------
    y_true : np.ndarray
        Binary default labels.
    scores_dict : dict[str, np.ndarray]
        Model name -> predicted PD array.
    save_path : str
        File path for the saved figure.
    """
    plt.style.use(_STYLE)
    y_clean, clean = _aligned_arrays(y_true, scores_dict)
    ar_scores = compute_cap_ar(y_true, scores_dict)

    n = len(y_clean)
    n_defaults = int(y_clean.sum())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Perfect model
    perfect_y = np.sort(y_clean)[::-1]
    cum_defaults_perfect = np.cumsum(perfect_y) / n_defaults
    x_axis = np.arange(1, n + 1) / n
    ax.plot(
        np.concatenate([[0], x_axis]),
        np.concatenate([[0], cum_defaults_perfect]),
        color="grey", lw=1.5, ls="--", label="Perfect Model",
    )

    # Random baseline
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random")

    # Model curves
    for name, pds in clean.items():
        order = np.argsort(-pds)
        y_sorted = y_clean[order]
        cum_defaults = np.cumsum(y_sorted) / n_defaults
        color = _COLORS.get(name, None)
        ax.plot(
            np.concatenate([[0], x_axis]),
            np.concatenate([[0], cum_defaults]),
            color=color, lw=2,
            label=f"{name} (AR = {ar_scores[name]:.3f})",
        )

    ax.set_xlabel("Cumulative % of Firms")
    ax.set_ylabel("Cumulative % of Defaults Captured")
    ax.set_title("CAP Curves \u2014 Default Discrimination")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI)
    plt.close(fig)


def plot_dd_distribution(
    panel_df: pd.DataFrame,
    save_path: str = "output/dd_distribution.png",
) -> None:
    """Plot distance-to-default histograms for Merton and VK models.

    Two subplots side by side, each showing overlapping histograms coloured
    by default label (defaulted vs non-defaulted).

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with columns: default_label, merton_DD, vk_DD.
    save_path : str
        File path for the saved figure.
    """
    plt.style.use(_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    dd_cols = [("merton_DD", "Merton DD"), ("vk_DD", "VK DD")]
    default_color = "#E53935"
    non_default_color = "#1E88E5"

    for ax, (col, title) in zip(axes, dd_cols):
        if col not in panel_df.columns:
            ax.set_title(f"{title} (data not available)")
            continue

        df = panel_df[[col, "default_label"]].dropna()
        defaulted = df.loc[df["default_label"] == 1, col]
        non_defaulted = df.loc[df["default_label"] == 0, col]

        bins = np.linspace(
            df[col].min() - 0.5,
            df[col].max() + 0.5,
            40,
        )

        ax.hist(non_defaulted, bins=bins, alpha=0.6, color=non_default_color,
                label="Non-defaulted", edgecolor="white", linewidth=0.5)
        ax.hist(defaulted, bins=bins, alpha=0.7, color=default_color,
                label="Defaulted", edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Distance to Default")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(frameon=True)

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI)
    plt.close(fig)


def plot_pd_comparison(
    panel_df: pd.DataFrame,
    save_path: str = "output/pd_comparison.png",
) -> None:
    """Pairwise scatter plots of predicted PDs across the three models.

    Three subplots: Merton vs VK, Merton vs HW, VK vs HW.
    Points are coloured by default label and a 45-degree reference line
    is shown on each subplot.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with columns: default_label, merton_PD, vk_PD, hw_PD.
    save_path : str
        File path for the saved figure.
    """
    plt.style.use(_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [
        ("merton_PD", "vk_PD", "Merton PD", "VK PD"),
        ("merton_PD", "hw_PD", "Merton PD", "HW PD"),
        ("vk_PD", "hw_PD", "VK PD", "HW PD"),
    ]

    default_color = "#E53935"
    non_default_color = "#1E88E5"

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        if xcol not in panel_df.columns or ycol not in panel_df.columns:
            ax.set_title(f"{xlabel} vs {ylabel} (data not available)")
            continue

        df = panel_df[[xcol, ycol, "default_label"]].dropna()
        non_def = df[df["default_label"] == 0]
        defaulted = df[df["default_label"] == 1]

        ax.scatter(non_def[xcol], non_def[ycol], c=non_default_color,
                   alpha=0.5, s=20, label="Non-defaulted", edgecolors="none")
        ax.scatter(defaulted[xcol], defaulted[ycol], c=default_color,
                   alpha=0.8, s=30, label="Defaulted", marker="x")

        # 45-degree reference line
        lo = min(df[xcol].min(), df[ycol].min(), 0)
        hi = max(df[xcol].max(), df[ycol].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.legend(fontsize=8, frameon=True)

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI)
    plt.close(fig)
