"""End-to-end training orchestration: data -> features -> model -> artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; no plt.show() anywhere in this codebase
import matplotlib.pyplot as plt
import numpy as np

from src.data.data_loader import discover_train_records, load_dataset_arrays, train_val_test_split
from src.data.synthetic_generator import generate_synthetic_dataset
from src.features.feature_extractor import ImageFeaturePipeline
from src.models.svm_model import SVMClassifier
from src.utils.config_loader import AppConfig, ensure_output_dirs, load_config
from src.utils.exceptions import PipelineError
from src.utils.logger import get_logger


def _save_confusion_matrix_figure(cm: list, out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cat", "dog"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["cat", "dog"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_training_pipeline(config_path: str = "config/config.yaml") -> dict:
    config: AppConfig = load_config(config_path)
    ensure_output_dirs(config)

    logger = get_logger(
        "pipeline.training",
        log_dir=config.get("paths", "logs_dir"),
        level=config.get("logging", "level", default="INFO"),
        max_bytes=config.get("logging", "max_bytes", default=1_048_576),
        backup_count=config.get("logging", "backup_count", default=3),
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("=== Training run %s started ===", run_id)

    try:
        if config.get("synthetic_data", "enabled_if_raw_missing", default=True):
            generate_synthetic_dataset(
                train_dir=config.get("paths", "train_dir"),
                num_images_per_class=config.get("synthetic_data", "num_images_per_class"),
                image_size=config.get("synthetic_data", "image_size"),
                noise_std=config.get("synthetic_data", "noise_std"),
                random_seed=config.get("project", "random_seed"),
            )

        records = discover_train_records(config.get("paths", "train_dir"))
        images, labels = load_dataset_arrays(
            records,
            resize_dim=config.get("image", "resize_dim"),
            color_mode=config.get("image", "color_mode"),
        )

        splits = train_val_test_split(
            images, labels,
            test_size=config.get("split", "test_size"),
            val_size=config.get("split", "val_size"),
            stratify=config.get("split", "stratify"),
            random_seed=config.get("project", "random_seed"),
        )

        feature_pipeline = ImageFeaturePipeline(
            hog_orientations=config.get("features", "hog", "orientations"),
            hog_pixels_per_cell=config.get("features", "hog", "pixels_per_cell"),
            hog_cells_per_block=config.get("features", "hog", "cells_per_block"),
            hog_block_norm=config.get("features", "hog", "block_norm"),
            hist_bins=config.get("features", "color_histogram", "bins_per_channel"),
        )

        x_train_feat = feature_pipeline.fit_transform(splits["x_train"])
        x_val_feat = feature_pipeline.transform(splits["x_val"])
        x_test_feat = feature_pipeline.transform(splits["x_test"])

        classifier = SVMClassifier(
            kernel=config.get("model", "kernel"),
            C=config.get("model", "C"),
            gamma=config.get("model", "gamma"),
            probability=config.get("model", "probability"),
            random_seed=config.get("project", "random_seed"),
        )
        classifier.fit(x_train_feat, splits["y_train"], grid_search=config.get("model", "grid_search"))

        val_metrics = classifier.evaluate(x_val_feat, splits["y_val"])
        test_metrics = classifier.evaluate(x_test_feat, splits["y_test"])

        model_dir = Path(config.get("paths", "model_dir"))
        metrics_dir = Path(config.get("paths", "metrics_dir"))
        figures_dir = Path(config.get("paths", "figures_dir"))

        classifier.save(model_dir / "svm_classifier.joblib")
        feature_pipeline.save(model_dir / "feature_pipeline.joblib")

        _save_confusion_matrix_figure(test_metrics["confusion_matrix"], figures_dir / "confusion_matrix.png")

        run_summary = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_train": int(len(splits["y_train"])),
            "n_val": int(len(splits["y_val"])),
            "n_test": int(len(splits["y_test"])),
            "feature_dim": int(x_train_feat.shape[1]),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "config_snapshot": config.raw,
        }

        with open(metrics_dir / "run_summary.json", "w", encoding="utf-8") as fh:
            json.dump(run_summary, fh, indent=2)

        logger.info("=== Training run %s complete. Test accuracy: %.4f ===",
                     run_id, test_metrics["accuracy"])
        return run_summary

    except PipelineError:
        logger.exception("Training pipeline failed with a domain error.")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training pipeline failed with an unexpected error.")
        raise PipelineError(f"Unexpected training failure: {exc}") from exc


if __name__ == "__main__":
    run_training_pipeline()
