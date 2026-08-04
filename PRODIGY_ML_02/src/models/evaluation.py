"""
Evaluation utilities for the clustering pipeline.

Computes silhouette score for the fixed-k model, an elbow curve across a
range of k values (informational -- the project's k stays fixed at 5 per
spec), and human-readable per-cluster profiles (mean Age/Income/Spending
Score, gender split, size) useful for business interpretation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.utils.exceptions import EvaluationError


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the silhouette score for a fitted clustering."""
    if len(set(labels)) < 2:
        raise EvaluationError("Silhouette score requires at least 2 distinct clusters.")
    try:
        return float(silhouette_score(X, labels))
    except Exception as exc:
        raise EvaluationError(f"Silhouette score computation failed: {exc}") from exc


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Davies-Bouldin index (lower is better) as a secondary metric."""
    try:
        return float(davies_bouldin_score(X, labels))
    except Exception as exc:
        raise EvaluationError(f"Davies-Bouldin score computation failed: {exc}") from exc


def compute_elbow_curve(
    X: np.ndarray,
    k_range: tuple[int, int],
    random_state: int,
    logger: logging.Logger | None = None,
) -> dict[int, float]:
    """
    Fit KMeans across a range of k values and record inertia for each, for
    the elbow plot. This is informational only -- it does not change the
    fixed n_clusters used by the production model.
    """
    k_min, k_max = k_range
    inertias: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km.fit(X)
        inertias[k] = float(km.inertia_)
    if logger:
        logger.info("Elbow curve computed for k in [%d, %d].", k_min, k_max)
    return inertias


def build_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Build human-readable per-cluster business profiles: size, mean feature
    values, and gender distribution where available.
    """
    profile_df = df.copy()
    profile_df["Cluster"] = labels

    profiles: dict[str, Any] = {}
    for cluster_id in sorted(profile_df["Cluster"].unique()):
        subset = profile_df[profile_df["Cluster"] == cluster_id]
        entry: dict[str, Any] = {
            "size": int(len(subset)),
            "pct_of_total": round(100 * len(subset) / len(profile_df), 1),
        }
        for col in feature_columns:
            entry[f"mean_{col}"] = round(float(subset[col].mean()), 2)

        if "Gender" in subset.columns:
            gender_counts = subset["Gender"].value_counts(normalize=True).round(3) * 100
            entry["gender_pct"] = {k: round(float(v), 1) for k, v in gender_counts.items()}

        profiles[f"cluster_{cluster_id}"] = entry

    if logger:
        logger.info("Built business profiles for %d clusters.", len(profiles))

    return profiles
