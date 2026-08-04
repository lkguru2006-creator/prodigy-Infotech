#!/usr/bin/env python3
"""
Sandbox-only test runner: exercises the same logic as tests/*.py via plain
Python assertions, since pytest cannot be installed in network-restricted
environments. This is NOT a replacement for the pytest suite -- in any
environment with pip access, run `pytest tests/` instead, which is the
authoritative, properly-structured test suite checked into tests/.

This script imports nothing from tests/ (pytest fixtures aren't usable
outside pytest) and instead re-derives equivalent sample data inline.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.loader import load_raw_data
from src.data.synthetic_data import generate_synthetic_customers
from src.features.feature_engineering import FeatureEngineer
from src.models.evaluation import build_cluster_profiles, compute_davies_bouldin, compute_elbow_curve, compute_silhouette
from src.models.kmeans_model import CustomerSegmentationModel
from src.pipeline.orchestrator import SegmentationPipeline
from src.utils.config_loader import load_config
from src.utils.exceptions import (
    ConfigurationError,
    DataLoadError,
    DataValidationError,
    FeatureEngineeringError,
    ModelTrainingError,
)
from src.utils.logger import get_logger

PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition: bool, description: str) -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {description}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {description}")


def check_raises(exc_type, fn, description: str) -> None:
    global PASS_COUNT, FAIL_COUNT
    try:
        fn()
        FAIL_COUNT += 1
        print(f"  [FAIL] {description} (no exception raised)")
    except exc_type:
        PASS_COUNT += 1
        print(f"  [PASS] {description}")
    except Exception as exc:  # noqa: BLE001
        FAIL_COUNT += 1
        print(f"  [FAIL] {description} (wrong exception: {type(exc).__name__}: {exc})")


def sample_customer_df(seed=7, n=40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "CustomerID": np.arange(1, n + 1),
            "Gender": rng.choice(["Male", "Female"], size=n),
            "Age": rng.integers(18, 70, size=n),
            "Annual Income (k$)": rng.integers(15, 140, size=n),
            "Spending Score (1-100)": rng.integers(1, 100, size=n),
        }
    )


def make_blobs(n_per_cluster=20, n_clusters=3, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(n_clusters, 2))
    X = np.vstack([rng.normal(c, 0.5, size=(n_per_cluster, 2)) for c in centers])
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    return X, labels


def main() -> int:
    import tempfile

    print("\n=== synthetic_data.py ===")
    df = generate_synthetic_customers(n_samples=100, n_blobs=5, random_state=42)
    check(len(df) == 100, "synthetic data has correct row count")
    check(
        list(df.columns)
        == ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"],
        "synthetic data has correct schema",
    )
    check(df["Age"].between(18, 70).all(), "synthetic Age within bounds")
    check(df["CustomerID"].is_unique, "synthetic CustomerID is unique")
    df1 = generate_synthetic_customers(n_samples=80, n_blobs=5, random_state=123)
    df2 = generate_synthetic_customers(n_samples=80, n_blobs=5, random_state=123)
    check(df1.equals(df2), "same seed produces reproducible synthetic data")
    check_raises(
        DataValidationError,
        lambda: generate_synthetic_customers(n_samples=2, n_blobs=5, random_state=42),
        "n_samples < n_blobs raises DataValidationError",
    )

    print("\n=== feature_engineering.py ===")
    fe = FeatureEngineer(
        feature_columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"], scale=True
    )
    cdf = sample_customer_df()
    X = fe.fit_transform(cdf)
    check(X.shape == (len(cdf), 3), "fit_transform output shape correct")
    check(np.allclose(X.mean(axis=0), 0, atol=1e-8), "scaled features have ~zero mean")
    check(np.allclose(X.std(axis=0), 1, atol=1e-8), "scaled features have unit variance")

    fe2 = FeatureEngineer(feature_columns=["Age"], scale=True)
    check_raises(
        FeatureEngineeringError,
        lambda: fe2.transform(cdf),
        "transform() before fit_transform() raises",
    )
    check_raises(
        FeatureEngineeringError,
        lambda: FeatureEngineer(feature_columns=["Nonexistent"], scale=True).fit_transform(cdf),
        "missing feature column raises",
    )
    check_raises(
        FeatureEngineeringError, lambda: FeatureEngineer(feature_columns=[], scale=True), "empty feature list raises"
    )

    print("\n=== kmeans_model.py ===")
    Xb, _ = make_blobs(n_clusters=3, seed=1)
    model = CustomerSegmentationModel(n_clusters=3, random_state=42)
    model.fit(Xb)
    check(model.cluster_centers_.shape == (3, 2), "cluster_centers_ has correct shape")
    check(model.labels_.shape == (Xb.shape[0],), "labels_ has correct shape")
    check(model.inertia_ > 0, "inertia_ is positive")
    check(np.array_equal(model.predict(Xb), model.labels_), "predict on training data matches labels_")

    check_raises(
        ModelTrainingError,
        lambda: CustomerSegmentationModel(n_clusters=3).fit(np.array([]).reshape(0, 2)),
        "fit on empty data raises",
    )
    check_raises(
        ModelTrainingError,
        lambda: CustomerSegmentationModel(n_clusters=5).fit(np.array([[1.0, 2.0], [3.0, 4.0]])),
        "fit with n_samples < n_clusters raises",
    )
    check_raises(
        ModelTrainingError,
        lambda: CustomerSegmentationModel(n_clusters=3).predict(np.array([[1.0, 2.0]])),
        "predict before fit raises",
    )

    m1 = CustomerSegmentationModel(n_clusters=4, random_state=42).fit(make_blobs(n_clusters=4, seed=1)[0])
    m2 = CustomerSegmentationModel(n_clusters=4, random_state=42).fit(make_blobs(n_clusters=4, seed=1)[0])
    check(np.isclose(m1.inertia_, m2.inertia_), "same seed gives reproducible inertia")

    print("\n=== evaluation.py ===")
    Xe, labels_e = make_blobs(n_clusters=3, seed=1)
    sil = compute_silhouette(Xe, labels_e)
    check(-1.0 <= sil <= 1.0, "silhouette score in valid range")
    check(sil > 0.5, "silhouette score high for well-separated blobs")
    check_raises(
        Exception,
        lambda: compute_silhouette(Xe, np.zeros(len(Xe), dtype=int)),
        "silhouette with single cluster raises",
    )
    db = compute_davies_bouldin(Xe, labels_e)
    check(isinstance(db, float) and db >= 0, "davies-bouldin returns non-negative float")
    inertias = compute_elbow_curve(Xe, k_range=(2, 6), random_state=42)
    check(set(inertias.keys()) == {2, 3, 4, 5, 6}, "elbow curve covers full k range")
    values = [inertias[k] for k in sorted(inertias)]
    check(all(values[i] >= values[i + 1] for i in range(len(values) - 1)), "elbow inertia is non-increasing")

    profiles = build_cluster_profiles(cdf, np.array([i % 3 for i in range(len(cdf))]), ["Age", "Annual Income (k$)"])
    check(set(profiles.keys()) == {"cluster_0", "cluster_1", "cluster_2"}, "cluster profiles have correct keys")
    check(sum(p["size"] for p in profiles.values()) == len(cdf), "cluster profile sizes sum to total")

    print("\n=== config_loader.py ===")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        config_dir = td_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        import yaml

        minimal_cfg = {
            "project": {"name": "t", "version": "0.0.1", "random_seed": 42},
            "data": {"raw_path": "data/raw/x.csv", "processed_path": "data/processed/y.csv"},
            "features": {},
            "model": {},
            "evaluation": {},
            "paths": {"model_dir": "outputs/models"},
            "logging": {},
        }
        config_file.write_text(yaml.dump(minimal_cfg))
        loaded = load_config(str(config_file))
        check(Path(loaded["data"]["raw_path"]).is_absolute(), "relative raw_path resolved to absolute")
        check(str(td_path) in loaded["data"]["raw_path"], "raw_path resolved relative to project root, not cwd")

        check_raises(
            ConfigurationError, lambda: load_config(str(td_path / "missing.yaml")), "missing config file raises"
        )

        bad_cfg = dict(minimal_cfg)
        del bad_cfg["model"]
        bad_file = config_dir / "bad.yaml"
        bad_file.write_text(yaml.dump(bad_cfg))
        check_raises(ConfigurationError, lambda: load_config(str(bad_file)), "missing required section raises")

    print("\n=== loader.py (data validation) ===")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        raw_path = td_path / "raw.csv"
        cdf2 = sample_customer_df()
        cdf2.to_csv(raw_path, index=False)

        cfg = {
            "data": {
                "raw_path": str(raw_path),
                "expected_columns": [
                    "CustomerID",
                    "Gender",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                ],
                "synthetic": {"enabled_if_missing": True, "n_samples": 60, "n_blobs": 5},
            },
            "project": {"random_seed": 42},
        }
        logger = get_logger("plain_test", log_dir=str(td_path / "logs"))
        loaded_df = load_raw_data(cfg, logger)
        check(len(loaded_df) == len(cdf2), "loader uses existing file when present")

        # Missing column case
        bad_df = cdf2.drop(columns=["Gender"])
        bad_path = td_path / "bad.csv"
        bad_df.to_csv(bad_path, index=False)
        cfg_bad = dict(cfg)
        cfg_bad["data"] = dict(cfg["data"])
        cfg_bad["data"]["raw_path"] = str(bad_path)
        check_raises(DataValidationError, lambda: load_raw_data(cfg_bad, logger), "missing column raises in loader")

        # Missing raw file, no synthetic fallback
        cfg_nofallback = dict(cfg)
        cfg_nofallback["data"] = dict(cfg["data"])
        cfg_nofallback["data"]["raw_path"] = str(td_path / "nope.csv")
        cfg_nofallback["data"]["synthetic"] = {"enabled_if_missing": False}
        check_raises(
            DataLoadError, lambda: load_raw_data(cfg_nofallback, logger), "missing file with no fallback raises"
        )

    print("\n=== full pipeline integration ===")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        full_cfg = {
            "project": {"name": "integration-test", "version": "0.0.1", "random_seed": 42},
            "data": {
                "raw_path": str(td_path / "data" / "raw" / "Mall_Customers.csv"),
                "processed_path": str(td_path / "data" / "processed" / "out.csv"),
                "expected_columns": [
                    "CustomerID",
                    "Gender",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                ],
                "synthetic": {"enabled_if_missing": True, "n_samples": 80, "n_blobs": 5},
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
                "model_dir": str(td_path / "outputs" / "models"),
                "metrics_dir": str(td_path / "outputs" / "metrics"),
                "plots_dir": str(td_path / "outputs" / "plots"),
                "log_dir": str(td_path / "outputs" / "logs"),
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
        logger = get_logger("integration_plain", log_dir=str(td_path / "outputs" / "logs"))
        pipeline = SegmentationPipeline(full_cfg, logger)
        summary = pipeline.run()

        check(summary["metrics"]["n_clusters"] == 5, "integration: n_clusters correct")
        check("silhouette_score" in summary["metrics"], "integration: silhouette present")
        check(len(summary["cluster_profiles"]) == 5, "integration: 5 cluster profiles built")
        check((td_path / "outputs" / "models" / "kmeans_model.joblib").exists(), "integration: model artifact saved")
        check((td_path / "outputs" / "models" / "scaler.joblib").exists(), "integration: scaler artifact saved")
        check((td_path / "outputs" / "plots" / "cluster_3d.png").exists(), "integration: 3D plot saved")
        check((td_path / "outputs" / "plots" / "elbow_curve.png").exists(), "integration: elbow plot saved")
        check(
            (td_path / "data" / "processed" / "out.csv").exists(), "integration: labeled dataset saved"
        )

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 50)
    return 1 if FAIL_COUNT > 0 else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
