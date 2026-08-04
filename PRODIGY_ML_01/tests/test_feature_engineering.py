"""Tests for FeatureEngineer."""

from __future__ import annotations

import pytest

from house_price_predictor.features.engineering import FeatureEngineer
from house_price_predictor.utils.exceptions import FeatureEngineeringError


def test_fit_transform_produces_engineered_columns(config, sample_train_df):
    fe = FeatureEngineer(config)
    out = fe.fit_transform(sample_train_df)
    for col in ("TotalSF", "TotalBath", "HouseAge"):
        assert col in out.columns


def test_fit_transform_imputes_missing_values(config, sample_train_df):
    fe = FeatureEngineer(config)
    out = fe.fit_transform(sample_train_df)
    assert out["TotalBsmtSF"].isna().sum() == 0


def test_transform_before_fit_raises(config, sample_raw_df):
    fe = FeatureEngineer(config)
    with pytest.raises(FeatureEngineeringError):
        fe.transform(sample_raw_df)


def test_transform_reuses_fitted_medians(config, sample_train_df, sample_raw_df):
    fe = FeatureEngineer(config)
    fe.fit_transform(sample_train_df)
    out = fe.transform(sample_raw_df)
    assert out["TotalBsmtSF"].isna().sum() == 0


def test_total_bath_calculation(config, sample_train_df):
    fe = FeatureEngineer(config)
    out = fe.fit_transform(sample_train_df)
    # row 0: FullBath=2, HalfBath=0 -> TotalBath=2.0
    row = out[out["Id"] == 1].iloc[0]
    assert row["TotalBath"] == pytest.approx(2.0)


def test_house_age_non_negative(config, sample_train_df):
    fe = FeatureEngineer(config)
    out = fe.fit_transform(sample_train_df)
    assert (out["HouseAge"] >= 0).all()


def test_missing_required_column_raises(config, sample_train_df):
    fe = FeatureEngineer(config)
    broken = sample_train_df.drop(columns=["GrLivArea"])
    with pytest.raises(FeatureEngineeringError):
        fe.fit_transform(broken)
