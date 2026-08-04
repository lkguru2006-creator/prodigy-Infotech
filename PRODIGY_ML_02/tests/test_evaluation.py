"""Tests for src.models.evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluation import (
    build_cluster_profiles,
    compute_davies_bouldin,
    compute_elbow_curve,
    compute_silhouette,
)
from src.utils.exceptions import EvaluationError


def _make_blobs(n_per_cluster=20, n_clusters=3, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(n_clusters, 2))
    X = np.vstack([rng.normal(c, 0.5, size=(n_per_cluster, 2)) for c in centers])
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    return X, labels


def test_compute_silhouette_well_separated_clusters_high_score():
    X, labels = _make_blobs(n_clusters=3, seed=1)
    score = compute_silhouette(X, labels)
    assert -1.0 <= score <= 1.0
    assert score > 0.5  # well-separated synthetic blobs should score well


def test_compute_silhouette_single_cluster_raises():
    X = np.random.default_rng(0).normal(size=(20, 2))
    labels = np.zeros(20, dtype=int)
    with pytest.raises(EvaluationError):
        compute_silhouette(X, labels)


def test_compute_davies_bouldin_returns_float():
    X, labels = _make_blobs(n_clusters=3, seed=2)
    score = compute_davies_bouldin(X, labels)
    assert isinstance(score, float)
    assert score >= 0


def test_compute_elbow_curve_returns_decreasing_inertia():
    X, _ = _make_blobs(n_clusters=4, seed=3)
    inertias = compute_elbow_curve(X, k_range=(2, 6), random_state=42)

    assert set(inertias.keys()) == {2, 3, 4, 5, 6}
    values = [inertias[k] for k in sorted(inertias)]
    # Inertia should be monotonically non-increasing as k grows
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def test_build_cluster_profiles_structure(sample_customer_df):
    labels = np.array([i % 3 for i in range(len(sample_customer_df))])
    profiles = build_cluster_profiles(
        sample_customer_df, labels, ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    )

    assert set(profiles.keys()) == {"cluster_0", "cluster_1", "cluster_2"}
    for profile in profiles.values():
        assert "size" in profile
        assert "mean_Age" in profile
        assert "gender_pct" in profile

    total_size = sum(p["size"] for p in profiles.values())
    assert total_size == len(sample_customer_df)
