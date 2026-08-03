"""Tests for DataValidator."""

from __future__ import annotations

import pandas as pd
import pytest

from house_price_predictor.data.validation import DataValidator
from house_price_predictor.utils.exceptions import DataValidationError


def test_valid_train_data_passes(config, sample_train_df):
    validator = DataValidator(config)
    validator.validate_raw(sample_train_df, is_train=True)  # should not raise


def test_empty_dataframe_fails(config):
    validator = DataValidator(config)
    with pytest.raises(DataValidationError):
        validator.validate_raw(pd.DataFrame(), is_train=True)


def test_missing_required_column_fails(config, sample_train_df):
    validator = DataValidator(config)
    broken = sample_train_df.drop(columns=["GrLivArea"])
    with pytest.raises(DataValidationError):
        validator.validate_raw(broken, is_train=True)


def test_missing_target_on_train_fails(config, sample_train_df):
    validator = DataValidator(config)
    broken = sample_train_df.drop(columns=["SalePrice"])
    with pytest.raises(DataValidationError):
        validator.validate_raw(broken, is_train=True)


def test_negative_sale_price_fails(config, sample_train_df):
    validator = DataValidator(config)
    broken = sample_train_df.copy()
    broken.loc[0, "SalePrice"] = -100
    with pytest.raises(DataValidationError):
        validator.validate_raw(broken, is_train=True)


def test_negative_bedrooms_fails(config, sample_train_df):
    validator = DataValidator(config)
    broken = sample_train_df.copy()
    broken.loc[0, "BedroomAbvGr"] = -1
    with pytest.raises(DataValidationError):
        validator.validate_raw(broken, is_train=True)


def test_test_set_does_not_require_target(config, sample_raw_df):
    validator = DataValidator(config)
    validator.validate_raw(sample_raw_df, is_train=False)  # should not raise
