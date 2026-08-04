"""
Inference.

Loads a previously trained model and applies it to new, raw input data,
running the same feature engineering used at training time to guarantee
consistent feature representation.
"""

from __future__ import annotations

import pandas as pd

from house_price_predictor.features.engineering import FeatureEngineer
from house_price_predictor.models.persistence import ModelPersistence
from house_price_predictor.utils.config import AppConfig
from house_price_predictor.utils.exceptions import InferenceError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceService:
    """Serves predictions for new house records using saved model artifacts."""

    def __init__(self, config: AppConfig, feature_engineer: FeatureEngineer):
        """
        Parameters
        ----------
        config : AppConfig
        feature_engineer : FeatureEngineer
            Must already be fitted (i.e. produced via the training pipeline)
            so imputation statistics match what the model was trained on.
        """
        self.config = config
        self.feature_engineer = feature_engineer
        self.persistence = ModelPersistence(config)
        self._estimator = None
        self._scaler = None
        self._feature_names: list[str] | None = None

    def load_model(self) -> None:
        """Load the persisted estimator/scaler/feature list into memory."""
        try:
            self._estimator, self._scaler, self._feature_names = self.persistence.load()
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Could not load model for inference: {exc}") from exc

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for raw input records.

        Parameters
        ----------
        raw_df : DataFrame containing the id column and required raw features
                 (not yet feature-engineered).

        Returns
        -------
        DataFrame with [id_column, 'PredictedSalePrice']
        """
        if self._estimator is None or self._scaler is None or self._feature_names is None:
            raise InferenceError("Model not loaded. Call load_model() before predict().")

        try:
            id_col = self.config.id_column
            if id_col not in raw_df.columns:
                raise InferenceError(f"Input data missing id column '{id_col}'")

            engineered = self.feature_engineer.transform(raw_df)
            X = engineered[self._feature_names]
            X_scaled = self._scaler.transform(X.values)

            predictions = self._estimator.predict(X_scaled)

            result = pd.DataFrame(
                {
                    id_col: raw_df[id_col].values,
                    "PredictedSalePrice": predictions,
                }
            )
            logger.info("Generated %d predictions.", len(result))
            return result

        except InferenceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Prediction failed: {exc}") from exc
