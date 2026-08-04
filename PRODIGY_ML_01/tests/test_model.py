"""Tests for ModelTrainer and ModelEvaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from house_price_predictor.models.evaluator import ModelEvaluator
from house_price_predictor.models.trainer import ModelTrainer
from house_price_predictor.utils.exceptions import ModelTrainingError


@pytest.fixture
def toy_xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=50),
            "b": rng.normal(size=50),
        }
    )
    y = pd.Series(3 * X["a"] + 2 * X["b"] + 10 + rng.normal(scale=0.01, size=50))
    return X, y


def test_train_returns_fitted_model(config, toy_xy):
    X, y = toy_xy
    trainer = ModelTrainer(config)
    model = trainer.train(X, y)
    assert model.estimator is not None
    assert model.feature_names == ["a", "b"]


def test_train_runs_cross_validation(config, toy_xy):
    X, y = toy_xy
    trainer = ModelTrainer(config)
    model = trainer.train(X, y)
    assert model.cv_rmse_mean is not None
    assert model.cv_rmse_mean >= 0


def test_train_on_empty_data_raises(config):
    trainer = ModelTrainer(config)
    with pytest.raises(ModelTrainingError):
        trainer.train(pd.DataFrame(), pd.Series(dtype=float))


def test_train_mismatched_lengths_raises(config):
    trainer = ModelTrainer(config)
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2])
    with pytest.raises(ModelTrainingError):
        trainer.train(X, y)


def test_evaluator_perfect_prediction():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    result = ModelEvaluator.evaluate(y_true, y_pred)
    assert result.rmse == pytest.approx(0.0)
    assert result.mae == pytest.approx(0.0)
    assert result.r2 == pytest.approx(1.0)
    assert result.mape == pytest.approx(0.0)


def test_evaluator_shape_mismatch_raises():
    with pytest.raises(ModelTrainingError):
        ModelEvaluator.evaluate(np.array([1.0, 2.0]), np.array([1.0]))
