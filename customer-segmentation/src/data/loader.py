"""
Data loading module with schema validation and synthetic-data fallback.

If the real Kaggle CSV is present at data/raw/Mall_Customers.csv, it is
used directly. If absent, a synthetic dataset matching the exact same
schema is generated and persisted to the same path, so the pipeline is
runnable out of the box and the synthetic file can simply be deleted
and replaced with the real one later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.synthetic_data import generate_synthetic_customers
from src.utils.exceptions import DataLoadError, DataValidationError


def load_raw_data(config: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Load the raw customer dataset, generating synthetic data if the real
    file is not present and synthetic fallback is enabled in config.

    Args:
        config: parsed pipeline configuration.
        logger: structured logger instance.

    Returns:
        Validated raw DataFrame.

    Raises:
        DataLoadError: if the file is missing and synthetic fallback is
            disabled, or if the file cannot be read.
        DataValidationError: if the loaded data fails schema validation.
    """
    data_cfg = config["data"]
    raw_path = Path(data_cfg["raw_path"])

    if not raw_path.exists():
        if not data_cfg.get("synthetic", {}).get("enabled_if_missing", False):
            raise DataLoadError(
                f"Raw data file not found at '{raw_path}' and synthetic "
                "fallback is disabled in config."
            )
        logger.warning(
            "Raw data file not found at '%s'. Generating synthetic data "
            "matching the Kaggle Mall Customers schema as a placeholder.",
            raw_path,
        )
        synthetic_cfg = data_cfg["synthetic"]
        df = generate_synthetic_customers(
            n_samples=synthetic_cfg["n_samples"],
            n_blobs=synthetic_cfg["n_blobs"],
            random_state=config["project"]["random_seed"],
            logger=logger,
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        logger.info("Synthetic data persisted to '%s' for reproducibility.", raw_path)
    else:
        try:
            df = pd.read_csv(raw_path)
            logger.info("Loaded raw data from '%s' (%d rows).", raw_path, len(df))
        except Exception as exc:
            raise DataLoadError(f"Failed to read CSV at '{raw_path}': {exc}") from exc

    _validate_schema(df, data_cfg["expected_columns"], logger)
    return df


def _validate_schema(
    df: pd.DataFrame, expected_columns: list[str], logger: logging.Logger
) -> None:
    """
    Validate that the loaded DataFrame has the expected columns, no fully
    empty rows, and no duplicate customer IDs.

    Raises:
        DataValidationError: on any schema or quality violation.
    """
    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Dataset is missing expected column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    if df.empty:
        raise DataValidationError("Loaded dataset is empty.")

    if df["CustomerID"].duplicated().any():
        n_dupes = int(df["CustomerID"].duplicated().sum())
        raise DataValidationError(f"Dataset contains {n_dupes} duplicate CustomerID values.")

    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    for col in numeric_cols:
        if df[col].isna().any():
            n_missing = int(df[col].isna().sum())
            raise DataValidationError(
                f"Column '{col}' contains {n_missing} missing value(s)."
            )
        if (df[col] < 0).any():
            raise DataValidationError(f"Column '{col}' contains negative value(s).")

    logger.info(
        "Schema validation passed: %d rows, %d columns, no missing values, no duplicate IDs.",
        len(df),
        len(df.columns),
    )
