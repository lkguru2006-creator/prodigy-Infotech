"""
End-to-end integration test for the full segmentation pipeline.

Runs the orchestrator against an isolated tmp_path directory (synthetic
data only, no real Kaggle file needed) and asserts that every expected
artifact is produced, the run summary is well-formed, and a second run
reproduces identical results given the same seed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.pipeline.orchestrator import SegmentationPipeline

_logger = logging.getLogger("integration_test")
_logger.addHandler(logging.NullHandler())


def _build_isolated_config(tmp_path: Path, sample_config: dict) -> dict:
    """Rewrite all paths in sample_config to live under tmp_path."""
    cfg = json.loads(json.dumps(sample_config))  # deep copy
    cfg["data"]["raw_path"] = str(tmp_path / "data" / "raw" / "Mall_Customers.csv")
    cfg["data"]["processed_path"] = str(tmp_path / "data" / "processed" / "customers_clustered.csv")
    cfg["paths"]["model_dir"] = str(tmp_path / "outputs" / "models")
    cfg["paths"]["metrics_dir"] = str(tmp_path / "outputs" / "metrics")
    cfg["paths"]["plots_dir"] = str(tmp_path / "outputs" / "plots")
    cfg["paths"]["log_dir"] = str(tmp_path / "outputs" / "logs")
    return cfg


def test_full_pipeline_produces_all_artifacts(tmp_path: Path, sample_config: dict):
    cfg = _build_isolated_config(tmp_path, sample_config)
    pipeline = SegmentationPipeline(cfg, _logger)

    summary = pipeline.run()

    assert summary["metrics"]["n_clusters"] == 5
    assert "silhouette_score" in summary["metrics"]
    assert len(summary["cluster_profiles"]) == 5

    # Model artifacts
    assert (tmp_path / "outputs" / "models" / "kmeans_model.joblib").exists()
    assert (tmp_path / "outputs" / "models" / "scaler.joblib").exists()
    assert (tmp_path / "outputs" / "models" / "feature_list.json").exists()

    # Metrics artifacts
    assert (tmp_path / "outputs" / "metrics" / "metrics.json").exists()
    assert (tmp_path / "outputs" / "metrics" / "cluster_profiles.json").exists()
    assert (tmp_path / "outputs" / "metrics" / "run_summary.json").exists()

    # Plots
    plots_dir = tmp_path / "outputs" / "plots"
    assert (plots_dir / "cluster_sizes.png").exists()
    assert (plots_dir / "cluster_scatter_2d.png").exists()
    assert (plots_dir / "cluster_pairplot.png").exists()
    assert (plots_dir / "cluster_3d.png").exists()
    assert (plots_dir / "elbow_curve.png").exists()

    # Labeled dataset
    processed_path = tmp_path / "data" / "processed" / "customers_clustered.csv"
    assert processed_path.exists()


def test_full_pipeline_reproducible_across_runs(tmp_path: Path, sample_config: dict):
    cfg1 = _build_isolated_config(tmp_path / "run1", sample_config)
    cfg2 = _build_isolated_config(tmp_path / "run2", sample_config)

    summary1 = SegmentationPipeline(cfg1, _logger).run()
    summary2 = SegmentationPipeline(cfg2, _logger).run()

    assert summary1["metrics"]["inertia"] == summary2["metrics"]["inertia"]
    assert summary1["metrics"]["silhouette_score"] == summary2["metrics"]["silhouette_score"]


def test_full_pipeline_logs_no_errors_on_happy_path(tmp_path: Path, sample_config: dict, caplog):
    cfg = _build_isolated_config(tmp_path, sample_config)
    with caplog.at_level(logging.ERROR):
        SegmentationPipeline(cfg, _logger).run()

    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
