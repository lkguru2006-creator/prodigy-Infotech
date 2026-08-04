"""
Data ingestion.

Responsible for getting raw train/test data into memory. Prefers real
Kaggle CSV files (data/raw/train.csv, data/raw/test.csv) if present;
otherwise falls back to the synthetic generator so the pipeline always
remains runnable end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from house_price_predictor.data.synthetic_generator import generate_synthetic_dataset
from house_price_predictor.utils.config import AppConfig
from house_price_predictor.utils.exceptions import DataIngestionError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Loads raw train/test data, from disk if available, else synthesizes it."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.raw_dir: Path = config.path("data", "raw_dir")
        self.train_filename = config.get("data", "train_file", default="train.csv")
        self.test_filename = config.get("data", "test_file", default="test.csv")

    @property
    def train_path(self) -> Path:
        return self.raw_dir / self.train_filename

    @property
    def test_path(self) -> Path:
        return self.raw_dir / self.test_filename

    def _real_data_available(self) -> bool:
        return self.train_path.exists() and self.test_path.exists()

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw train and test DataFrames.

        Returns
        -------
        (train_df, test_df)
        """
        try:
            if self._real_data_available():
                logger.info("Loading real dataset from %s", self.raw_dir)
                train_df = pd.read_csv(self.train_path)
                test_df = pd.read_csv(self.test_path)
                logger.info(
                    "Loaded real data: train=%s, test=%s", train_df.shape, test_df.shape
                )
                return train_df, test_df

            logger.warning(
                "Real Kaggle CSVs not found at %s — falling back to synthetic data "
                "generator. Place train.csv/test.csv there to use real data.",
                self.raw_dir,
            )
            synth_cfg = self.config.get("data", "synthetic", default={})
            n_train = int(synth_cfg.get("n_train_samples", 1460))
            n_test = int(synth_cfg.get("n_test_samples", 1459))

            train_df, test_df = generate_synthetic_dataset(
                n_train=n_train, n_test=n_test, seed=self.config.random_seed
            )

            self.raw_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(self.train_path, index=False)
            test_df.to_csv(self.test_path, index=False)
            logger.info("Persisted synthetic data to %s", self.raw_dir)

            return train_df, test_df

        except DataIngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"Data ingestion failed: {exc}") from exc
