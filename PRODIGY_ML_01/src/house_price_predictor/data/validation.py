"""
Data validation.

Schema and sanity checks applied immediately after ingestion, before any
feature engineering occurs. Fails fast with a clear, specific error rather
than letting bad data silently propagate through the pipeline.
"""

from __future__ import annotations

import pandas as pd

from house_price_predictor.utils.config import AppConfig
from house_price_predictor.utils.exceptions import DataValidationError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validates raw input DataFrames against expected schema and ranges."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.required_numeric = list(config.get("features", "numeric_features", default=[]))
        self.target_column = config.target_column
        self.id_column = config.id_column

    def validate_raw(self, df: pd.DataFrame, *, is_train: bool) -> None:
        """
        Validate a raw DataFrame fresh from ingestion.

        Raises
        ------
        DataValidationError
            If required columns are missing, the frame is empty, or
            critical columns contain impossible values.
        """
        label = "train" if is_train else "test"

        if df.empty:
            raise DataValidationError(f"{label} dataset is empty.")

        missing_cols = [c for c in self.required_numeric if c not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"{label} dataset is missing required columns: {missing_cols}"
            )

        if self.id_column not in df.columns:
            raise DataValidationError(f"{label} dataset is missing id column '{self.id_column}'")

        if is_train and self.target_column not in df.columns:
            raise DataValidationError(
                f"Train dataset is missing target column '{self.target_column}'"
            )

        if is_train:
            target = df[self.target_column]
            if (target <= 0).any():
                raise DataValidationError(
                    f"Train dataset contains non-positive '{self.target_column}' values."
                )
            if target.isna().any():
                raise DataValidationError(
                    f"Train dataset contains missing '{self.target_column}' values."
                )

        # Sanity bounds on key numeric features (won't hard-fail on minor
        # missingness, only on structurally impossible values)
        if "BedroomAbvGr" in df.columns and (df["BedroomAbvGr"] < 0).any():
            raise DataValidationError("Negative bedroom counts detected.")

        if "FullBath" in df.columns and (df["FullBath"] < 0).any():
            raise DataValidationError("Negative bathroom counts detected.")

        logger.info(
            "%s dataset passed validation: shape=%s, columns=%d",
            label.capitalize(),
            df.shape,
            len(df.columns),
        )
