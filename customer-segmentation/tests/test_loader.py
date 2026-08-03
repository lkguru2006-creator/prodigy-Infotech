"""Tests for src.data.loader."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_raw_data
from src.utils.exceptions import DataLoadError, DataValidationError

_logger = logging.getLogger("test")
_logger.addHandler(logging.NullHandler())


def test_load_raw_data_generates_synthetic_when_missing(tmp_path: Path, sample_config: dict):
    sample_config["data"]["raw_path"] = str(tmp_path / "raw" / "Mall_Customers.csv")

    df = load_raw_data(sample_config, _logger)

    assert len(df) == sample_config["data"]["synthetic"]["n_samples"]
    assert Path(sample_config["data"]["raw_path"]).exists()


def test_load_raw_data_uses_existing_file_when_present(
    tmp_path: Path, sample_config: dict, sample_customer_df: pd.DataFrame
):
    raw_path = tmp_path / "raw" / "Mall_Customers.csv"
    raw_path.parent.mkdir(parents=True)
    sample_customer_df.to_csv(raw_path, index=False)
    sample_config["data"]["raw_path"] = str(raw_path)

    df = load_raw_data(sample_config, _logger)

    assert len(df) == len(sample_customer_df)


def test_load_raw_data_missing_file_no_fallback_raises(tmp_path: Path, sample_config: dict):
    sample_config["data"]["raw_path"] = str(tmp_path / "raw" / "Mall_Customers.csv")
    sample_config["data"]["synthetic"]["enabled_if_missing"] = False

    with pytest.raises(DataLoadError):
        load_raw_data(sample_config, _logger)


def test_load_raw_data_missing_column_raises(tmp_path: Path, sample_config: dict, sample_customer_df: pd.DataFrame):
    raw_path = tmp_path / "raw" / "Mall_Customers.csv"
    raw_path.parent.mkdir(parents=True)
    sample_customer_df.drop(columns=["Gender"]).to_csv(raw_path, index=False)
    sample_config["data"]["raw_path"] = str(raw_path)

    with pytest.raises(DataValidationError, match="missing expected column"):
        load_raw_data(sample_config, _logger)


def test_load_raw_data_duplicate_ids_raise(tmp_path: Path, sample_config: dict, sample_customer_df: pd.DataFrame):
    raw_path = tmp_path / "raw" / "Mall_Customers.csv"
    raw_path.parent.mkdir(parents=True)
    df = sample_customer_df.copy()
    df.loc[1, "CustomerID"] = df.loc[0, "CustomerID"]
    df.to_csv(raw_path, index=False)
    sample_config["data"]["raw_path"] = str(raw_path)

    with pytest.raises(DataValidationError, match="duplicate"):
        load_raw_data(sample_config, _logger)


def test_load_raw_data_missing_values_raise(tmp_path: Path, sample_config: dict, sample_customer_df: pd.DataFrame):
    raw_path = tmp_path / "raw" / "Mall_Customers.csv"
    raw_path.parent.mkdir(parents=True)
    df = sample_customer_df.copy()
    df.loc[0, "Age"] = None
    df.to_csv(raw_path, index=False)
    sample_config["data"]["raw_path"] = str(raw_path)

    with pytest.raises(DataValidationError, match="missing value"):
        load_raw_data(sample_config, _logger)


def test_load_raw_data_empty_file_raises(tmp_path: Path, sample_config: dict, sample_customer_df: pd.DataFrame):
    raw_path = tmp_path / "raw" / "Mall_Customers.csv"
    raw_path.parent.mkdir(parents=True)
    sample_customer_df.iloc[0:0].to_csv(raw_path, index=False)
    sample_config["data"]["raw_path"] = str(raw_path)

    with pytest.raises(DataValidationError, match="empty"):
        load_raw_data(sample_config, _logger)
