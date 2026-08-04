"""
Feature engineering for the clustering pipeline.

Follows fit/transform separation: scaling parameters are learned once via
`fit()` and reused via `transform()`, mirroring scikit-learn convention.
This is good practice even though K-means here is fit once on the full
dataset (no train/test split, since clustering is unsupervised), because
it keeps the scaler reusable for scoring any new customer record later
(e.g. assigning a new customer to an existing cluster).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.exceptions import FeatureEngineeringError


class FeatureEngineer:
    """
    Encapsulates feature selection and scaling for clustering.

    Usage:
        fe = FeatureEngineer(feature_columns=[...], scale=True)
        X_scaled = fe.fit_transform(df)
        ...
        X_new_scaled = fe.transform(new_df)   # reuses fitted scaler
    """

    def __init__(self, feature_columns: list[str], scale: bool = True):
        if not feature_columns:
            raise FeatureEngineeringError("feature_columns must be a non-empty list.")
        self.feature_columns = feature_columns
        self.scale = scale
        self.scaler: StandardScaler | None = None
        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame, logger: logging.Logger | None = None) -> np.ndarray:
        """Fit the scaler on df and return the transformed feature matrix."""
        X = self._select_features(df)

        if self.scale:
            self.scaler = StandardScaler()
            X_transformed = self.scaler.fit_transform(X)
            if logger:
                logger.info(
                    "Fitted StandardScaler on features %s. Mean=%s, Scale=%s",
                    self.feature_columns,
                    np.round(self.scaler.mean_, 2).tolist(),
                    np.round(self.scaler.scale_, 2).tolist(),
                )
        else:
            X_transformed = X.to_numpy()

        self._is_fitted = True
        return X_transformed

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the already-fitted scaler."""
        if not self._is_fitted:
            raise FeatureEngineeringError(
                "FeatureEngineer.transform() called before fit_transform(). "
                "Fit on training data first to avoid train/test leakage."
            )
        X = self._select_features(df)
        if self.scale and self.scaler is not None:
            return self.scaler.transform(X)
        return X.to_numpy()

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise FeatureEngineeringError(
                f"Cannot select features; missing column(s): {missing}"
            )
        return df[self.feature_columns].copy()

    def save_feature_list(self, path: str | Path) -> None:
        """Persist the feature list as JSON for downstream reproducibility/auditing."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"feature_columns": self.feature_columns, "scaled": self.scale},
                f,
                indent=2,
            )
