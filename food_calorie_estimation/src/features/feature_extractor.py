"""Image feature extraction: HOG texture descriptors + color histograms.

Fit/transform are strictly separated: `fit` only computes the StandardScaler
statistics on the training set; `transform` applies them. This prevents test-set
statistics from leaking into training, matching the project's established pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

from src.utils.exceptions import FeatureExtractionError


class FoodFeatureExtractor:
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        hog_cfg = cfg["features"]["hog"]
        self.hog_orientations = hog_cfg["orientations"]
        self.hog_pixels_per_cell = tuple(hog_cfg["pixels_per_cell"])
        self.hog_cells_per_block = tuple(hog_cfg["cells_per_block"])
        self.color_bins = cfg["features"]["color_hist_bins"]
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _extract_single(self, image: np.ndarray) -> np.ndarray:
        gray = rgb2gray(image)
        hog_features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            feature_vector=True,
        )
        color_hist = np.concatenate(
            [
                np.histogram(image[:, :, ch], bins=self.color_bins, range=(0, 255))[0]
                for ch in range(3)
            ]
        ).astype(np.float32)
        color_hist = color_hist / (color_hist.sum() + 1e-8)
        return np.concatenate([hog_features, color_hist])

    def _extract_batch(self, images: np.ndarray) -> np.ndarray:
        try:
            return np.stack([self._extract_single(img) for img in images], axis=0)
        except Exception as exc:  # noqa: BLE001 - wrap any skimage failure uniformly
            raise FeatureExtractionError(f"Failed to extract features: {exc}") from exc

    def fit(self, images: np.ndarray) -> "FoodFeatureExtractor":
        raw_features = self._extract_batch(images)
        self.scaler.fit(raw_features)
        self._is_fitted = True
        self.logger.info("Feature extractor fitted on %d images (%d-dim raw features)",
                          len(images), raw_features.shape[1])
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise FeatureExtractionError("FoodFeatureExtractor.transform called before fit().")
        raw_features = self._extract_batch(images)
        return self.scaler.transform(raw_features)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        self.fit(images)
        return self.transform(images)
