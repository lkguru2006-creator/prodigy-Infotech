"""
Prediction pipeline orchestrator.

Loads the saved model and generates predictions for the held-out Kaggle
test set (the one without ground-truth SalePrice), writing a submission
file in the Kaggle-expected format.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from house_price_predictor.data.ingestion import DataIngestion
from house_price_predictor.features.engineering import FeatureEngineer
from house_price_predictor.models.inference import InferenceService
from house_price_predictor.utils.config import AppConfig
from house_price_predictor.utils.exceptions import HousePricePredictorError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionPipeline:
    """Generates a submission-ready predictions file using the saved model."""

    def __init__(self, config: AppConfig, fitted_feature_engineer: FeatureEngineer):
        """
        Parameters
        ----------
        config : AppConfig
        fitted_feature_engineer : FeatureEngineer
            Must be the SAME fitted instance used during training, so
            imputation medians and feature ordering match exactly.
        """
        self.config = config
        self.ingestion = DataIngestion(config)
        self.inference_service = InferenceService(config, fitted_feature_engineer)

    def run(self, output_filename: str = "submission.csv") -> Path:
        """Run inference on the test set and write a submission CSV."""
        try:
            logger.info("=== Prediction pipeline started ===")
            self.inference_service.load_model()

            _, test_df = self.ingestion.load()
            predictions = self.inference_service.predict(test_df)

            submission = predictions.rename(
                columns={"PredictedSalePrice": self.config.target_column}
            )

            reports_dir = self.config.path("paths", "reports_dir")
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = reports_dir / output_filename
            submission.to_csv(output_path, index=False)

            logger.info("Predictions written to %s (%d rows)", output_path, len(submission))
            logger.info("=== Prediction pipeline completed successfully ===")
            return output_path

        except HousePricePredictorError:
            logger.exception("Prediction pipeline failed with a known pipeline error.")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction pipeline failed with an unexpected error.")
            raise HousePricePredictorError(f"Unexpected prediction failure: {exc}") from exc
