"""
Visualization utilities for the clustering pipeline.

All plots are generated with matplotlib's non-interactive 'Agg' backend
and saved directly to disk -- no plt.show() calls, no blocking windows,
no stray figures left open. Each function closes its figure explicitly
to keep memory clean across repeated pipeline runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  -- must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

_CLUSTER_PALETTE = "viridis"


def plot_elbow_curve(inertias: dict[int, float], output_path: str | Path) -> None:
    """Plot inertia vs. k for the elbow method (informational)."""
    ks = list(inertias.keys())
    values = list(inertias.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, values, marker="o", linewidth=2, color="#4C72B0")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_title("Elbow Method for Optimal k (informational; production k is fixed at 5)")
    ax.set_xticks(ks)
    fig.tight_layout()

    _save_and_close(fig, output_path)


def plot_cluster_scatter(
    df: pd.DataFrame,
    labels: np.ndarray,
    x_col: str,
    y_col: str,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """2D scatter of two features, colored by cluster assignment."""
    plot_df = df.copy()
    plot_df["Cluster"] = labels.astype(str)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue="Cluster",
        palette=_CLUSTER_PALETTE,
        s=70,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title(title or f"Customer Segments: {x_col} vs {y_col}")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    _save_and_close(fig, output_path)


def plot_cluster_pairplot(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    output_path: str | Path,
) -> None:
    """Pairwise feature relationships colored by cluster, for full-dataset overview."""
    plot_df = df[feature_columns].copy()
    plot_df["Cluster"] = labels.astype(str)

    grid = sns.pairplot(
        plot_df,
        hue="Cluster",
        palette=_CLUSTER_PALETTE,
        diag_kind="kde",
        plot_kws={"alpha": 0.75, "s": 40},
    )
    grid.fig.suptitle("Cluster Pairplot Across All Clustering Features", y=1.02)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)


def plot_cluster_3d(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    output_path: str | Path,
) -> None:
    """3D scatter of the three clustering features, colored by cluster."""
    if len(feature_columns) != 3:
        return  # 3D plot only meaningful for exactly 3 features; skip silently otherwise

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = (df[col].to_numpy() for col in feature_columns)
    scatter = ax.scatter(x, y, z, c=labels, cmap=_CLUSTER_PALETTE, s=50, alpha=0.85)

    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.set_zlabel(feature_columns[2])
    ax.set_title("3D Customer Segmentation View")

    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left")
    ax.add_artist(legend)
    fig.tight_layout()

    _save_and_close(fig, output_path)


def plot_cluster_sizes(
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Bar chart of customer count per cluster."""
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette(_CLUSTER_PALETTE, n_colors=len(unique))
    ax.bar([str(u) for u in unique], counts, color=colors)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count per Cluster")
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.01, str(c), ha="center", fontweight="bold")
    fig.tight_layout()

    _save_and_close(fig, output_path)


def _save_and_close(fig: plt.Figure, output_path: str | Path) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
