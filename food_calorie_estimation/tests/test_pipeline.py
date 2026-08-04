"""Pytest test suite. Run with `pytest tests/` in environments with network access
(pip install -r requirements.txt). For sandboxed/offline environments, use
tests/verify_pipeline.py instead, which contains equivalent checks as plain asserts.
"""
import numpy as np
import pytest

from src.utils.config_loader import load_config
from src.features.feature_extractor import FoodFeatureExtractor
from src.models.calorie_estimator import CalorieEstimator
from src.models.sklearn_classifier import SklearnFoodClassifier
from src.utils.logger import get_logger


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def logger(cfg):
    return get_logger("test_logger", cfg)


def test_config_paths_are_absolute(cfg):
    assert cfg["data"]["raw_dir"].startswith("/")
    assert cfg["outputs"]["models_dir"].startswith("/")


def test_calorie_estimator_known_class(cfg, logger):
    estimator = CalorieEstimator(cfg, logger)
    assert estimator.estimate("pizza") == 266


def test_calorie_estimator_unknown_class_falls_back_to_default(cfg, logger):
    estimator = CalorieEstimator(cfg, logger)
    assert estimator.estimate("not_a_real_class") == cfg["calorie_lookup"]["default"]


def test_feature_extractor_fit_transform_shape(cfg, logger):
    fe = FoodFeatureExtractor(cfg, logger)
    fake_images = np.random.randint(0, 255, size=(5, 64, 64, 3), dtype=np.uint8)
    features = fe.fit_transform(fake_images)
    assert features.shape[0] == 5


def test_feature_extractor_raises_before_fit(cfg, logger):
    fe = FoodFeatureExtractor(cfg, logger)
    fake_images = np.random.randint(0, 255, size=(2, 64, 64, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        fe.transform(fake_images)


def test_classifier_raises_before_fit(cfg, logger):
    clf = SklearnFoodClassifier(cfg, logger)
    with pytest.raises(Exception):
        clf.predict(np.zeros((1, 10)))
