"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from house_price_predictor.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    project_root = Path(__file__).resolve().parents[1]
    return load_config(project_root / "config" / "config.yaml")


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5],
            "GrLivArea": [1500.0, 1800.0, 1200.0, 2200.0, 1600.0],
            "TotalBsmtSF": [800.0, 900.0, None, 1100.0, 700.0],
            "BedroomAbvGr": [3, 4, 2, 5, 3],
            "FullBath": [2, 2, 1, 3, 2],
            "HalfBath": [0, 1, 0, 1, 0],
            "OverallQual": [6, 7, 5, 9, 6],
            "YearBuilt": [2000, 2010, 1995, 2018, 2005],
            "GarageCars": [2, 2, 1, 3, 2],
        }
    )


@pytest.fixture
def sample_train_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_raw_df.copy()
    df["SalePrice"] = [210000, 260000, 150000, 340000, 225000]
    return df
