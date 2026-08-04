"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_config() -> dict:
    """A minimal, valid config dict for unit tests that don't need the full YAML."""
    return {
        "project": {"name": "test-project", "version": "0.0.1", "random_seed": 42},
        "data": {
            "raw_path": "data/raw/Mall_Customers.csv",
            "processed_path": "data/processed/customers_clustered.csv",
            "expected_columns": [
                "CustomerID",
                "Gender",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)",
            ],
            "id_column": "CustomerID",
            "synthetic": {"enabled_if_missing": True, "n_samples": 60, "n_blobs": 5},
        },
        "features": {
            "clustering_features": ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
            "scale_features": True,
        },
        "model": {
            "algorithm": "kmeans",
            "n_clusters": 5,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42,
        },
        "evaluation": {
            "compute_silhouette": True,
            "compute_elbow_curve": True,
            "elbow_k_range": [2, 6],
            "generate_pairplot": True,
            "generate_3d_plot": True,
        },
        "paths": {
            "model_dir": "outputs/models",
            "metrics_dir": "outputs/metrics",
            "plots_dir": "outputs/plots",
            "log_dir": "outputs/logs",
            "model_filename": "kmeans_model.joblib",
            "scaler_filename": "scaler.joblib",
            "feature_list_filename": "feature_list.json",
            "metrics_filename": "metrics.json",
            "run_summary_filename": "run_summary.json",
            "cluster_profile_filename": "cluster_profiles.json",
            "log_filename": "pipeline.log",
        },
        "logging": {
            "level": "INFO",
            "max_bytes": 1048576,
            "backup_count": 3,
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        },
    }


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """A small, valid customer DataFrame matching the Mall Customers schema."""
    rng = np.random.default_rng(7)
    n = 40
    return pd.DataFrame(
        {
            "CustomerID": np.arange(1, n + 1),
            "Gender": rng.choice(["Male", "Female"], size=n),
            "Age": rng.integers(18, 70, size=n),
            "Annual Income (k$)": rng.integers(15, 140, size=n),
            "Spending Score (1-100)": rng.integers(1, 100, size=n),
        }
    )
