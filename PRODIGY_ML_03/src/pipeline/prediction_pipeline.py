"""Inference orchestration: load persisted artifacts, score new images,
write a Kaggle-format submission CSV (id,label where label=1 means dog).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data.data_loader import load_image_as_array
from src.features.feature_extractor import ImageFeaturePipeline
from src.models.svm_model import SVMClassifier
from src.utils.config_loader import AppConfig, ensure_output_dirs, load_config
from src.utils.exceptions import PipelineError, PredictionError
from src.utils.logger import get_logger


def run_prediction_pipeline(
    input_dir: str,
    config_path: str = "config/config.yaml",
    output_csv_name: str = "submission.csv",
) -> Path:
    config: AppConfig = load_config(config_path)
    ensure_output_dirs(config)

    logger = get_logger(
        "pipeline.prediction",
        log_dir=config.get("paths", "logs_dir"),
        level=config.get("logging", "level", default="INFO"),
    )

    model_dir = Path(config.get("paths", "model_dir"))
    outputs_dir = Path(config.get("paths", "outputs_dir"))

    try:
        classifier = SVMClassifier.load(model_dir / "svm_classifier.joblib")
        feature_pipeline = ImageFeaturePipeline.load(model_dir / "feature_pipeline.joblib")

        image_paths = sorted(Path(input_dir).glob("*.jpg"))
        if not image_paths:
            raise PredictionError(f"No .jpg images found in {input_dir}")

        resize_dim = config.get("image", "resize_dim")
        color_mode = config.get("image", "color_mode")

        images = np.stack([
            load_image_as_array(p, resize_dim, color_mode) for p in image_paths
        ])
        features = feature_pipeline.transform(images)
        predictions = classifier.predict(features)

        out_path = outputs_dir / output_csv_name
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "label"])
            for path, pred in zip(image_paths, predictions):
                writer.writerow([path.stem, int(pred)])

        logger.info("Wrote %d predictions to %s", len(predictions), out_path)
        return out_path

    except PipelineError:
        logger.exception("Prediction pipeline failed with a domain error.")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction pipeline failed with an unexpected error.")
        raise PredictionError(f"Unexpected prediction failure: {exc}") from exc
