"""Tests for src.features.feature_engineering."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_engineering import FeatureEngineer
from src.utils.exceptions import FeatureEngineeringError


def test_fit_transform_returns_scaled_array(sample_customer_df):
    fe = FeatureEngineer(
        feature_columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"], scale=True
    )
    X = fe.fit_transform(sample_customer_df)

    assert X.shape == (len(sample_customer_df), 3)
    # Scaled features should have ~zero mean and unit variance
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1, atol=1e-8)


def test_fit_transform_without_scaling_returns_raw_values(sample_customer_df):
    fe = FeatureEngineer(feature_columns=["Age"], scale=False)
    X = fe.fit_transform(sample_customer_df)

    assert np.array_equal(X.flatten(), sample_customer_df["Age"].to_numpy())


def test_transform_before_fit_raises(sample_customer_df):
    fe = FeatureEngineer(feature_columns=["Age"], scale=True)
    with pytest.raises(FeatureEngineeringError, match="before fit_transform"):
        fe.transform(sample_customer_df)


def test_transform_reuses_fitted_scaler_no_leakage(sample_customer_df):
    fe = FeatureEngineer(feature_columns=["Age", "Annual Income (k$)"], scale=True)
    fe.fit_transform(sample_customer_df)

    # New data with a different distribution should be transformed using
    # the ORIGINAL fitted scaler's mean/scale, not refit.
    new_df = sample_customer_df.copy()
    new_df["Age"] = new_df["Age"] + 1000  # wildly different distribution

    X_new = fe.transform(new_df)
    # Since scaler wasn't refit, this should NOT have zero mean
    assert not np.allclose(X_new.mean(axis=0), 0, atol=1)


def test_missing_feature_column_raises(sample_customer_df):
    fe = FeatureEngineer(feature_columns=["NonexistentColumn"], scale=True)
    with pytest.raises(FeatureEngineeringError, match="missing column"):
        fe.fit_transform(sample_customer_df)


def test_empty_feature_list_raises():
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineer(feature_columns=[], scale=True)


def test_save_feature_list_writes_json(tmp_path, sample_customer_df):
    fe = FeatureEngineer(feature_columns=["Age", "Annual Income (k$)"], scale=True)
    fe.fit_transform(sample_customer_df)

    out_path = tmp_path / "feature_list.json"
    fe.save_feature_list(out_path)

    assert out_path.exists()
