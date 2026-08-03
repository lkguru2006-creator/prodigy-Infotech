"""
K-means clustering model wrapper.

Wraps scikit-learn's KMeans with config-driven hyperparameters, structured
logging, and consistent error handling, matching the pattern used in the
regression pipeline's model module.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from src.utils.exceptions import ModelTrainingError


class CustomerSegmentationModel:
    """Thin, config-driven wrapper around sklearn's KMeans."""

    def __init__(
        self,
        n_clusters: int = 5,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, logger: logging.Logger | None = None) -> "CustomerSegmentationModel":
        """Fit K-means on the (already scaled) feature matrix."""
        if X is None or len(X) == 0:
            raise ModelTrainingError("Cannot fit model on empty feature matrix.")
        if len(X) < self.n_clusters:
            raise ModelTrainingError(
                f"n_samples ({len(X)}) must be >= n_clusters ({self.n_clusters})."
            )

        try:
            self.model.fit(X)
            self._is_fitted = True
        except Exception as exc:
            raise ModelTrainingError(f"KMeans fitting failed: {exc}") from exc

        if logger:
            logger.info(
                "KMeans fitted: k=%d, inertia=%.2f, n_iter=%d",
                self.n_clusters,
                self.model.inertia_,
                self.model.n_iter_,
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster labels to (already scaled) feature rows."""
        if not self._is_fitted:
            raise ModelTrainingError("predict() called before fit().")
        return self.model.predict(X)

    @property
    def cluster_centers_(self) -> np.ndarray:
        if not self._is_fitted:
            raise ModelTrainingError("cluster_centers_ accessed before fit().")
        return self.model.cluster_centers_

    @property
    def inertia_(self) -> float:
        if not self._is_fitted:
            raise ModelTrainingError("inertia_ accessed before fit().")
        return float(self.model.inertia_)

    @property
    def labels_(self) -> np.ndarray:
        if not self._is_fitted:
            raise ModelTrainingError("labels_ accessed before fit().")
        return self.model.labels_
