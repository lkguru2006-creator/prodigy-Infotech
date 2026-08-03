"""
Model training.

Wraps scikit-learn's LinearRegression with feature scaling and an
optional cross-validation diagnostic. Keeps the trained estimator and
scaler together as a single cohesive object for downstream persistence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from house_price_predictor.utils.config import AppConfig
from house_price_predictor.utils.exceptions import ModelTrainingError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    """Container bundling a fitted estimator with its fitted scaler."""

    estimator: LinearRegression
    scaler: StandardScaler
    feature_names: list[str]
    cv_rmse_mean: float | None = None
    cv_rmse_std: float | None = None


class ModelTrainer:
    """Trains a (scaled) linear regression model with optional cross-validation."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.scale_features = bool(config.get("model", "scale_features", default=True))
        self.cv_enabled = bool(
            config.get("model", "cross_validation", "enabled", default=True)
        )
        self.n_splits = int(config.get("model", "cross_validation", "n_splits", default=5))
        self.random_seed = config.random_seed

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        """
        Fit a linear regression model on (X, y).

        Parameters
        ----------
        X : feature matrix (already engineered)
        y : target vector (SalePrice)

        Returns
        -------
        TrainedModel
        """
        try:
            if X.empty or y.empty:
                raise ModelTrainingError("Cannot train on an empty feature matrix or target.")
            if len(X) != len(y):
                raise ModelTrainingError(
                    f"Feature/target length mismatch: X={len(X)} rows, y={len(y)} rows."
                )

            feature_names = list(X.columns)
            scaler = StandardScaler()

            if self.scale_features:
                X_scaled = scaler.fit_transform(X.values)
            else:
                X_scaled = X.values

            estimator = LinearRegression()

            cv_rmse_mean: float | None = None
            cv_rmse_std: float | None = None

            if self.cv_enabled and len(X) >= self.n_splits:
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
                neg_mse_scores = cross_val_score(
                    estimator, X_scaled, y.values, cv=kf, scoring="neg_mean_squared_error"
                )
                rmse_scores = np.sqrt(-neg_mse_scores)
                cv_rmse_mean = float(rmse_scores.mean())
                cv_rmse_std = float(rmse_scores.std())
                logger.info(
                    "Cross-validation RMSE: %.2f (+/- %.2f) across %d folds",
                    cv_rmse_mean,
                    cv_rmse_std,
                    self.n_splits,
                )

            estimator.fit(X_scaled, y.values)
            logger.info(
                "Model trained on %d samples with %d features.", len(X), len(feature_names)
            )

            return TrainedModel(
                estimator=estimator,
                scaler=scaler,
                feature_names=feature_names,
                cv_rmse_mean=cv_rmse_mean,
                cv_rmse_std=cv_rmse_std,
            )

        except ModelTrainingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelTrainingError(f"Model training failed: {exc}") from exc
