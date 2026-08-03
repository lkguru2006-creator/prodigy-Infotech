"""CPU-only classifier backend built on scikit-learn.

Selected via config.model.backend ("sklearn_rf" or "sklearn_mlp"). This class
implements BaseFoodClassifier, so swapping in a real CNN backend later
(e.g. src/models/keras_cnn_model.py) requires no changes to the pipeline layer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.models.base_model import BaseFoodClassifier
from src.utils.exceptions import ModelError


class SklearnFoodClassifier(BaseFoodClassifier):
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.backend_name = cfg["model"]["backend"]
        self.model = self._build_model()
        self._is_fitted = False

    def _build_model(self):
        if self.backend_name == "sklearn_rf":
            params = self.cfg["model"]["random_forest"]
            return RandomForestClassifier(**params)
        if self.backend_name == "sklearn_mlp":
            params = self.cfg["model"]["mlp"]
            return MLPClassifier(**params)
        raise ModelError(f"Unknown model backend: {self.backend_name}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnFoodClassifier":
        try:
            self.model.fit(X, y)
            self._is_fitted = True
            self.logger.info("Model '%s' fitted on %d samples", self.backend_name, len(X))
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Model training failed: {exc}") from exc
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info("Model saved to %s", path)

    def load(self, path: Path) -> "SklearnFoodClassifier":
        if not path.exists():
            raise ModelError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise ModelError("Model.predict called before fit()/load().")
