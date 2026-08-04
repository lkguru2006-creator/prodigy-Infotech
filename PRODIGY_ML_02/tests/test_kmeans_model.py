"""Tests for src.models.kmeans_model."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.kmeans_model import CustomerSegmentationModel
from src.utils.exceptions import ModelTrainingError


def _make_blobs(n_per_cluster=20, n_clusters=3, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(n_clusters, 2))
    X = np.vstack([rng.normal(c, 0.5, size=(n_per_cluster, 2)) for c in centers])
    return X


def test_fit_sets_fitted_attributes():
    X = _make_blobs(n_clusters=3)
    model = CustomerSegmentationModel(n_clusters=3, random_state=42)
    model.fit(X)

    assert model.cluster_centers_.shape == (3, 2)
    assert model.labels_.shape == (X.shape[0],)
    assert model.inertia_ > 0


def test_fit_on_empty_data_raises():
    model = CustomerSegmentationModel(n_clusters=3)
    with pytest.raises(ModelTrainingError, match="empty"):
        model.fit(np.array([]).reshape(0, 2))


def test_fit_with_fewer_samples_than_clusters_raises():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    model = CustomerSegmentationModel(n_clusters=5)
    with pytest.raises(ModelTrainingError, match="n_samples"):
        model.fit(X)


def test_predict_before_fit_raises():
    model = CustomerSegmentationModel(n_clusters=3)
    with pytest.raises(ModelTrainingError, match="before fit"):
        model.predict(np.array([[1.0, 2.0]]))


def test_predict_assigns_known_clusters_correctly():
    X = _make_blobs(n_per_cluster=30, n_clusters=3, seed=7)
    model = CustomerSegmentationModel(n_clusters=3, random_state=7)
    model.fit(X)

    # Predicting on the same training data should reproduce the same labels
    preds = model.predict(X)
    assert np.array_equal(preds, model.labels_)


def test_same_seed_gives_reproducible_results():
    X = _make_blobs(n_clusters=4, seed=1)
    m1 = CustomerSegmentationModel(n_clusters=4, random_state=42).fit(X)
    m2 = CustomerSegmentationModel(n_clusters=4, random_state=42).fit(X)

    assert np.isclose(m1.inertia_, m2.inertia_)
