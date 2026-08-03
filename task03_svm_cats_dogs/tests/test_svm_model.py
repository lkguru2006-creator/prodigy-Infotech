"""Pytest-syntax tests for SVMClassifier."""
import numpy as np
import pytest

from src.models.svm_model import SVMClassifier
from src.utils.exceptions import ModelError


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (20, 5)), rng.normal(5, 1, (20, 5))])
    y = np.array([0] * 20 + [1] * 20)
    return x, y


def test_fit_predict(toy_data):
    x, y = toy_data
    clf = SVMClassifier()
    clf.fit(x, y)
    preds = clf.predict(x)
    assert preds.shape == y.shape


def test_predict_before_fit_raises():
    clf = SVMClassifier()
    with pytest.raises(ModelError):
        clf.predict(np.zeros((2, 5)))


def test_evaluate_returns_expected_keys(toy_data):
    x, y = toy_data
    clf = SVMClassifier()
    clf.fit(x, y)
    metrics = clf.evaluate(x, y)
    for key in ("accuracy", "precision", "recall", "f1_score", "confusion_matrix"):
        assert key in metrics
