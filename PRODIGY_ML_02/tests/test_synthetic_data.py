"""Tests for src.data.synthetic_data."""

from __future__ import annotations

import pytest

from src.data.synthetic_data import generate_synthetic_customers
from src.utils.exceptions import DataValidationError


def test_generate_synthetic_customers_shape_and_columns():
    df = generate_synthetic_customers(n_samples=100, n_blobs=5, random_state=42)

    assert len(df) == 100
    assert list(df.columns) == [
        "CustomerID",
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
    ]


def test_generate_synthetic_customers_value_bounds():
    df = generate_synthetic_customers(n_samples=200, n_blobs=5, random_state=42)

    assert df["Age"].between(18, 70).all()
    assert df["Annual Income (k$)"].between(15, 140).all()
    assert df["Spending Score (1-100)"].between(1, 100).all()
    assert set(df["Gender"].unique()).issubset({"Male", "Female"})


def test_generate_synthetic_customers_unique_ids():
    df = generate_synthetic_customers(n_samples=150, n_blobs=5, random_state=42)
    assert df["CustomerID"].is_unique


def test_generate_synthetic_customers_reproducible_with_same_seed():
    df1 = generate_synthetic_customers(n_samples=80, n_blobs=5, random_state=123)
    df2 = generate_synthetic_customers(n_samples=80, n_blobs=5, random_state=123)
    assert df1.equals(df2)


def test_generate_synthetic_customers_different_seeds_differ():
    df1 = generate_synthetic_customers(n_samples=80, n_blobs=5, random_state=1)
    df2 = generate_synthetic_customers(n_samples=80, n_blobs=5, random_state=2)
    assert not df1["Age"].equals(df2["Age"])


def test_generate_synthetic_customers_invalid_params_raise():
    with pytest.raises(DataValidationError):
        generate_synthetic_customers(n_samples=2, n_blobs=5, random_state=42)


def test_generate_synthetic_customers_handles_nonstandard_blob_count():
    df = generate_synthetic_customers(n_samples=50, n_blobs=3, random_state=42)
    assert len(df) == 50
