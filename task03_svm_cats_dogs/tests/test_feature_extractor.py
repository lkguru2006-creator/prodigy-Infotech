"""Pytest-syntax tests for ImageFeaturePipeline (fit/transform separation)."""
import numpy as np
import pytest

from src.features.feature_extractor import ImageFeaturePipeline
from src.utils.exceptions import FeatureExtractionError


@pytest.fixture
def dummy_images():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(6, 64, 64, 3), dtype=np.uint8)


def test_fit_transform_shapes(dummy_images):
    pipeline = ImageFeaturePipeline()
    features = pipeline.fit_transform(dummy_images)
    assert features.shape[0] == dummy_images.shape[0]
    assert features.ndim == 2


def test_transform_before_fit_raises(dummy_images):
    pipeline = ImageFeaturePipeline()
    with pytest.raises(FeatureExtractionError):
        pipeline.transform(dummy_images)


def test_scaler_not_refit_on_transform(dummy_images):
    pipeline = ImageFeaturePipeline()
    pipeline.fit(dummy_images[:4])
    mean_before = pipeline.scaler.mean_.copy()
    pipeline.transform(dummy_images[4:])
    assert np.array_equal(mean_before, pipeline.scaler.mean_)
