"""End-to-end integration test for the training pipeline using synthetic data."""

from __future__ import annotations

from house_price_predictor.pipeline.training_pipeline import TrainingPipeline


def test_training_pipeline_end_to_end(config, tmp_path, monkeypatch):
    # Redirect data/artifact paths to a temp directory so tests never touch
    # the real project's data/artifacts.
    monkeypatch.setattr(config, "project_root", tmp_path, raising=False)

    pipeline = TrainingPipeline(config)
    result = pipeline.run()

    assert result.trained_model.estimator is not None
    assert result.holdout_metrics.r2 is not None
    assert result.holdout_metrics.rmse >= 0
    assert len(result.trained_model.feature_names) > 0
