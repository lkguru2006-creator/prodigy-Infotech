"""
Model persistence.

Handles saving and loading the trained estimator, fitted scaler, feature
list, and evaluation metrics as versioned artifacts on disk, so trained
models can be reused for inference without retraining.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from house_price_predictor.models.trainer import TrainedModel
from house_price_predictor.utils.config import AppConfig
from house_price_predictor.utils.exceptions import ModelPersistenceError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPersistence:
    """Saves/loads model artifacts to/from the configured artifacts directory."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models_dir: Path = config.path("paths", "models_dir")
        self.reports_dir: Path = config.path("paths", "reports_dir")
        self.model_filename = config.get(
            "model", "model_filename", default="linear_regression_model.joblib"
        )
        self.scaler_filename = config.get(
            "model", "scaler_filename", default="feature_scaler.joblib"
        )
        self.feature_list_filename = config.get(
            "model", "feature_list_filename", default="feature_list.json"
        )
        self.metrics_filename = config.get("model", "metrics_filename", default="metrics.json")

    def save(self, trained_model: TrainedModel, metrics: dict) -> None:
        """Persist estimator, scaler, feature list, and metrics to disk."""
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(trained_model.estimator, self.models_dir / self.model_filename)
            joblib.dump(trained_model.scaler, self.models_dir / self.scaler_filename)

            with open(self.models_dir / self.feature_list_filename, "w", encoding="utf-8") as fh:
                json.dump(trained_model.feature_names, fh, indent=2)

            full_metrics = dict(metrics)
            full_metrics["cv_rmse_mean"] = trained_model.cv_rmse_mean
            full_metrics["cv_rmse_std"] = trained_model.cv_rmse_std

            with open(self.reports_dir / self.metrics_filename, "w", encoding="utf-8") as fh:
                json.dump(full_metrics, fh, indent=2)

            logger.info("Model artifacts saved to %s", self.models_dir)

        except Exception as exc:  # noqa: BLE001
            raise ModelPersistenceError(f"Failed to save model artifacts: {exc}") from exc

    def load(self) -> tuple[object, object, list[str]]:
        """
        Load a previously saved estimator, scaler, and feature list.

        Returns
        -------
        (estimator, scaler, feature_names)
        """
        try:
            model_path = self.models_dir / self.model_filename
            scaler_path = self.models_dir / self.scaler_filename
            features_path = self.models_dir / self.feature_list_filename

            for path in (model_path, scaler_path, features_path):
                if not path.exists():
                    raise ModelPersistenceError(f"Expected artifact not found: {path}")

            estimator = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            with open(features_path, "r", encoding="utf-8") as fh:
                feature_names = json.load(fh)

            logger.info("Model artifacts loaded from %s", self.models_dir)
            return estimator, scaler, feature_names

        except ModelPersistenceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelPersistenceError(f"Failed to load model artifacts: {exc}") from exc
