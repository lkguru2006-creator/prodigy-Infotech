"""SVM model wrapper: train, evaluate, persist, and reload."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.utils.exceptions import ModelError
from src.utils.logger import get_logger

logger = get_logger("models.svm_model")


class SVMClassifier:
    """Thin, persistence-aware wrapper around sklearn's SVC."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 5.0,
        gamma: str | float = "scale",
        probability: bool = True,
        random_seed: int = 42,
    ):
        self.model = SVC(
            kernel=kernel, C=C, gamma=gamma, probability=probability,
            random_state=random_seed,
        )
        self._is_fitted = False

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        grid_search: dict[str, Any] | None = None,
    ) -> "SVMClassifier":
        try:
            if grid_search and grid_search.get("enabled"):
                search = GridSearchCV(
                    self.model,
                    param_grid=grid_search["param_grid"],
                    cv=grid_search.get("cv", 3),
                    n_jobs=-1,
                )
                search.fit(x_train, y_train)
                self.model = search.best_estimator_
                logger.info("Grid search complete. Best params: %s", search.best_params_)
            else:
                self.model.fit(x_train, y_train)
            self._is_fitted = True
            logger.info("SVM trained on %d samples, %d features", *x_train.shape)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"SVM training failed: {exc}") from exc
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ModelError("predict() called before the model was fitted.")
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ModelError("predict_proba() called before the model was fitted.")
        return self.model.predict_proba(x)

    def evaluate(self, x: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
        y_pred = self.predict(x)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "n_samples": int(len(y_true)),
        }
        logger.info(
            "Evaluation -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
            metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"],
        )
        return metrics

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))
        logger.info("Model persisted to %s", path)

    @staticmethod
    def load(path: str | Path) -> "SVMClassifier":
        obj = joblib.load(Path(path))
        if not isinstance(obj, SVMClassifier):
            raise ModelError(f"Loaded object at {path} is not an SVMClassifier")
        return obj
