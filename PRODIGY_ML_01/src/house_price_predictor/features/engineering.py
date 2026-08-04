"""
Feature engineering.

Transforms raw columns into the final feature set used for modeling:
  - imputes missing numeric values (median strategy)
  - engineers TotalSF, TotalBath, HouseAge
  - clips extreme outliers using an IQR-based rule
  - exposes a stable, ordered feature list for downstream consistency

This module is stateful: `fit_transform` learns imputation values from the
training set and stores them; `transform` reuses those learned values for
the test/inference set, preventing train/test leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from house_price_predictor.utils.config import AppConfig
from house_price_predictor.utils.exceptions import FeatureEngineeringError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureEngineer:
    """Learns and applies feature transformations consistently across splits."""

    config: AppConfig
    _median_values: dict[str, float] = field(default_factory=dict, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.base_numeric = list(
            self.config.get("features", "numeric_features", default=[])
        )
        self.reference_year = int(
            self.config.get("features", "reference_year", default=2026)
        )
        self.iqr_multiplier = float(
            self.config.get("features", "outlier_iqr_multiplier", default=3.0)
        )
        self.final_feature_order = self.base_numeric + list(
            self.config.get("features", "engineered_features", default=[])
        )

    # ---- public API -------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn imputation stats from `df`, then transform it."""
        try:
            self._median_values = {
                col: float(df[col].median()) for col in self.base_numeric if col in df.columns
            }
            self._is_fitted = True
            logger.info("FeatureEngineer fitted on %d rows.", len(df))
            return self._transform(df, clip_outliers=True)
        except Exception as exc:  # noqa: BLE001
            raise FeatureEngineeringError(f"fit_transform failed: {exc}") from exc

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously learned transformations to new data."""
        if not self._is_fitted:
            raise FeatureEngineeringError(
                "FeatureEngineer.transform() called before fit_transform()."
            )
        try:
            return self._transform(df, clip_outliers=False)
        except Exception as exc:  # noqa: BLE001
            raise FeatureEngineeringError(f"transform failed: {exc}") from exc

    @property
    def feature_names(self) -> list[str]:
        return list(self.final_feature_order)

    # ---- internals ----------------------------------------------------------

    def _transform(self, df: pd.DataFrame, *, clip_outliers: bool) -> pd.DataFrame:
        out = df.copy()

        # 1. Impute missing numeric values using learned (train-set) medians
        for col in self.base_numeric:
            if col not in out.columns:
                raise FeatureEngineeringError(f"Expected column '{col}' not present in data.")
            if out[col].isna().any():
                fill_value = self._median_values.get(col, out[col].median())
                out[col] = out[col].fillna(fill_value)

        # 2. Engineered features
        out["TotalSF"] = out["GrLivArea"] + out["TotalBsmtSF"]
        out["TotalBath"] = out["FullBath"] + 0.5 * out["HalfBath"]
        out["HouseAge"] = (self.reference_year - out["YearBuilt"]).clip(lower=0)

        # 3. Outlier clipping (train only) via IQR rule on engineered TotalSF
        if clip_outliers:
            q1, q3 = out["TotalSF"].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            n_before = len(out)
            out = out[(out["TotalSF"] >= lower) & (out["TotalSF"] <= upper)].copy()
            n_clipped = n_before - len(out)
            if n_clipped > 0:
                logger.info("Removed %d outlier rows based on TotalSF IQR bounds.", n_clipped)

        # 4. Final column ordering — guarantees consistent feature matrix shape
        missing_final = [c for c in self.final_feature_order if c not in out.columns]
        if missing_final:
            raise FeatureEngineeringError(f"Missing engineered columns: {missing_final}")

        ordered_cols = [c for c in out.columns if c not in self.final_feature_order]
        out = out[ordered_cols + self.final_feature_order]

        # Replace any residual inf/-inf (defensive, e.g. from div-by-zero in future features)
        out = out.replace([np.inf, -np.inf], np.nan)
        for col in self.final_feature_order:
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())

        return out
