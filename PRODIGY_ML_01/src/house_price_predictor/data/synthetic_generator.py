"""
Synthetic data generator.

Produces a dataset with the same column names and realistic value ranges
as the real Kaggle "House Prices - Advanced Regression Techniques" dataset
(restricted to the columns this project actually uses, plus Id/target).

This exists ONLY as a stand-in for local development/testing when the real
train.csv / test.csv are not yet available. The moment the real Kaggle
files are placed in data/raw/, the DataIngestion module will prefer them
automatically and this generator is never invoked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from house_price_predictor.utils.exceptions import DataIngestionError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


def _generate_base_frame(n_samples: int, seed: int, id_offset: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    overall_qual = rng.integers(low=1, high=11, size=n_samples)
    year_built = rng.integers(low=1900, high=2023, size=n_samples)
    grliv_area = rng.normal(loc=1500, scale=500, size=n_samples).clip(334, None)
    total_bsmt_sf = rng.normal(loc=1050, scale=440, size=n_samples).clip(0, None)
    bedroom_abvgr = rng.integers(low=0, high=7, size=n_samples)
    full_bath = rng.integers(low=0, high=4, size=n_samples)
    half_bath = rng.integers(low=0, high=3, size=n_samples)
    garage_cars = rng.integers(low=0, high=5, size=n_samples)

    df = pd.DataFrame(
        {
            "Id": np.arange(id_offset, id_offset + n_samples),
            "GrLivArea": grliv_area.round(0),
            "TotalBsmtSF": total_bsmt_sf.round(0),
            "BedroomAbvGr": bedroom_abvgr,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "OverallQual": overall_qual,
            "YearBuilt": year_built,
            "GarageCars": garage_cars,
        }
    )

    # Inject a small amount of realistic missingness in basement sqft
    missing_mask = rng.random(n_samples) < 0.02
    df.loc[missing_mask, "TotalBsmtSF"] = np.nan

    return df


def _simulate_sale_price(df: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)

    total_sf = df["GrLivArea"].fillna(0) + df["TotalBsmtSF"].fillna(0)
    total_bath = df["FullBath"] + 0.5 * df["HalfBath"]
    house_age = 2026 - df["YearBuilt"]

    base_price = (
        45_000
        + 85 * total_sf
        + 12_000 * total_bath
        + 9_500 * df["BedroomAbvGr"]
        + 16_000 * df["OverallQual"]
        + 11_000 * df["GarageCars"]
        - 250 * house_age
    )

    noise = rng.normal(loc=0, scale=18_000, size=len(df))
    price = (base_price + noise).clip(lower=34_900)
    return price.round(0)


def generate_synthetic_dataset(
    n_train: int, n_test: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic train and test sets with a Kaggle-compatible schema.

    Returns
    -------
    (train_df, test_df) : tuple of DataFrames
        train_df includes the SalePrice target column; test_df does not.
    """
    try:
        logger.info(
            "Generating synthetic dataset (n_train=%d, n_test=%d, seed=%d)",
            n_train,
            n_test,
            seed,
        )

        train_df = _generate_base_frame(n_train, seed=seed, id_offset=1)
        train_df["SalePrice"] = _simulate_sale_price(train_df, seed=seed)

        test_df = _generate_base_frame(n_test, seed=seed + 999, id_offset=n_train + 1)

        logger.info(
            "Synthetic dataset generated: train shape=%s, test shape=%s",
            train_df.shape,
            test_df.shape,
        )
        return train_df, test_df

    except Exception as exc:  # noqa: BLE001
        raise DataIngestionError(f"Failed to generate synthetic dataset: {exc}") from exc
