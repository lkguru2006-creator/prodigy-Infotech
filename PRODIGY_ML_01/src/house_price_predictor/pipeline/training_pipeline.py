"""
Training pipeline orchestrator.

Wires together ingestion -> validation -> feature engineering -> train/test
split -> training -> evaluation -> persistence into a single, clean
end-to-end flow. This is the single entry point invoked by scripts/run_pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from house_price_predictor.data.ingestion import DataIngestion
from house_price_predictor.data.validation import DataValidator
from house_price_predictor.features.engineering import FeatureEngineer
from house_price_predictor.models.evaluator import EvaluationResult, ModelEvaluator
from house_price_predictor.models.persistence import ModelPersistence
from house_price_predictor.models.trainer import ModelTrainer, TrainedModel
from house_price_predictor.utils.config import AppConfig
from house_price_predictor.utils.exceptions import HousePricePredictorError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    trained_model: TrainedModel
    train_metrics: EvaluationResult
    holdout_metrics: EvaluationResult
    feature_engineer: FeatureEngineer


class TrainingPipeline:
    """End-to-end training pipeline for the house price linear regression model."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.ingestion = DataIngestion(config)
        self.validator = DataValidator(config)
        self.feature_engineer = FeatureEngineer(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator()
        self.persistence = ModelPersistence(config)

    def run(self) -> PipelineResult:
        """Execute the full training pipeline and return its results."""
        try:
            logger.info("=== Training pipeline started ===")

            # 1. Ingestion
            train_df, test_df = self.ingestion.load()

            # 2. Validation (fail fast on bad schema/data)
            self.validator.validate_raw(train_df, is_train=True)
            self.validator.validate_raw(test_df, is_train=False)

            # 3. Persist a copy of raw data snapshot for reproducibility
            self._save_processed_snapshot(train_df, is_train=True)
            self._save_processed_snapshot(test_df, is_train=False)

            # 4. Feature engineering (fit on train only — no leakage)
            target_col = self.config.target_column
            y = train_df[target_col]
            engineered_train = self.feature_engineer.fit_transform(train_df)
            # Re-align y to engineered_train index in case outlier rows were dropped
            y = y.loc[engineered_train.index]

            X = engineered_train[self.feature_engineer.feature_names]

            # 5. Train/holdout split
            test_size = float(self.config.get("model", "test_size", default=0.2))
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=test_size, random_state=self.config.random_seed
            )
            logger.info(
                "Split data: train=%d rows, holdout=%d rows", len(X_train), len(X_holdout)
            )

            # 6. Train model
            trained_model = self.trainer.train(X_train, y_train)

            # 7. Evaluate on train and holdout sets
            train_preds = self._predict_raw(trained_model, X_train)
            holdout_preds = self._predict_raw(trained_model, X_holdout)

            train_metrics = self.evaluator.evaluate(y_train.values, train_preds)
            holdout_metrics = self.evaluator.evaluate(y_holdout.values, holdout_preds)

            # 8. Persist artifacts
            metrics_payload = {
                "train": train_metrics.to_dict(),
                "holdout": holdout_metrics.to_dict(),
            }
            self.persistence.save(trained_model, metrics_payload)

            logger.info("=== Training pipeline completed successfully ===")

            return PipelineResult(
                trained_model=trained_model,
                train_metrics=train_metrics,
                holdout_metrics=holdout_metrics,
                feature_engineer=self.feature_engineer,
            )

        except HousePricePredictorError:
            logger.exception("Training pipeline failed with a known pipeline error.")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training pipeline failed with an unexpected error.")
            raise HousePricePredictorError(f"Unexpected pipeline failure: {exc}") from exc

    # ---- internals ----------------------------------------------------------

    @staticmethod
    def _predict_raw(trained_model: TrainedModel, X: pd.DataFrame):
        X_scaled = trained_model.scaler.transform(X.values)
        return trained_model.estimator.predict(X_scaled)

    def _save_processed_snapshot(self, df: pd.DataFrame, *, is_train: bool) -> None:
        processed_dir = self.config.path("data", "processed_dir")
        processed_dir.mkdir(parents=True, exist_ok=True)
        filename = self.config.get(
            "data",
            "processed_train_file" if is_train else "processed_test_file",
            default="train_processed.csv" if is_train else "test_processed.csv",
        )
        df.to_csv(processed_dir / filename, index=False)
