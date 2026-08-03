"""Image -> feature-vector extraction (HOG + color histogram) and scaling.

Fit/transform separation: ``StandardScaler`` statistics are fit ONLY on
the training partition inside :meth:`ImageFeaturePipeline.fit`, then
reused (never refit) for validation/test/inference data in
:meth:`ImageFeaturePipeline.transform`. This prevents train/test leakage.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

from src.utils.exceptions import FeatureExtractionError
from src.utils.logger import get_logger

logger = get_logger("features.feature_extractor")


def _extract_single(
    image: np.ndarray,
    hog_orientations: int,
    hog_pixels_per_cell: tuple[int, int],
    hog_cells_per_block: tuple[int, int],
    hog_block_norm: str,
    hist_bins: int,
) -> np.ndarray:
    try:
        gray = rgb2gray(image)
        hog_features = hog(
            gray,
            orientations=hog_orientations,
            pixels_per_cell=tuple(hog_pixels_per_cell),
            cells_per_block=tuple(hog_cells_per_block),
            block_norm=hog_block_norm,
            feature_vector=True,
        )

        hist_features = []
        for channel in range(image.shape[-1]):
            hist, _ = np.histogram(
                image[..., channel], bins=hist_bins, range=(0, 255), density=True
            )
            hist_features.append(hist)
        hist_features = np.concatenate(hist_features)

        return np.concatenate([hog_features, hist_features]).astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - re-raised as domain error
        raise FeatureExtractionError(f"Feature extraction failed: {exc}") from exc


class ImageFeaturePipeline:
    """Stateful HOG + color-histogram extractor with an internal scaler."""

    def __init__(
        self,
        hog_orientations: int = 9,
        hog_pixels_per_cell: tuple[int, int] = (8, 8),
        hog_cells_per_block: tuple[int, int] = (2, 2),
        hog_block_norm: str = "L2-Hys",
        hist_bins: int = 16,
    ):
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = tuple(hog_pixels_per_cell)
        self.hog_cells_per_block = tuple(hog_cells_per_block)
        self.hog_block_norm = hog_block_norm
        self.hist_bins = hist_bins
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _raw_features(self, images: np.ndarray) -> np.ndarray:
        return np.stack([
            _extract_single(
                img, self.hog_orientations, self.hog_pixels_per_cell,
                self.hog_cells_per_block, self.hog_block_norm, self.hist_bins,
            )
            for img in images
        ])

    def fit(self, images: np.ndarray) -> "ImageFeaturePipeline":
        """Extract raw features from TRAIN images only, then fit the scaler."""
        raw = self._raw_features(images)
        self.scaler.fit(raw)
        self._is_fitted = True
        logger.info("Feature pipeline fitted on %d images -> %d-dim feature vectors",
                     images.shape[0], raw.shape[1])
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Extract raw features and apply the already-fitted scaler."""
        if not self._is_fitted:
            raise FeatureExtractionError("transform() called before fit(); scaler not fitted.")
        raw = self._raw_features(images)
        return self.scaler.transform(raw)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        raw = self._raw_features(images)
        self.scaler.fit(raw)
        self._is_fitted = True
        return self.scaler.transform(raw)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))
        logger.info("Feature pipeline persisted to %s", path)

    @staticmethod
    def load(path: str | Path) -> "ImageFeaturePipeline":
        obj = joblib.load(Path(path))
        if not isinstance(obj, ImageFeaturePipeline):
            raise FeatureExtractionError(f"Loaded object at {path} is not an ImageFeaturePipeline")
        return obj
