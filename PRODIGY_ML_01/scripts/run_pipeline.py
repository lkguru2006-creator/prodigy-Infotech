#!/usr/bin/env python3
"""
Entry point: run the full training + prediction pipeline.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config/config.yaml
    python scripts/run_pipeline.py --skip-prediction

This is the only script meant to be invoked directly. It produces no raw
`print()` debug output — all progress is routed through the structured
logger (console + logs/pipeline.log), and final results are written to
artifacts/reports/ as JSON/CSV for downstream consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this script directly without installing the package.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from house_price_predictor.pipeline.prediction_pipeline import PredictionPipeline  # noqa: E402
from house_price_predictor.pipeline.training_pipeline import TrainingPipeline  # noqa: E402
from house_price_predictor.utils.config import load_config  # noqa: E402
from house_price_predictor.utils.exceptions import HousePricePredictorError  # noqa: E402
from house_price_predictor.utils.logger import get_logger, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="House Price Predictor pipeline runner.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to config/config.yaml at project root).",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Only train and evaluate; skip writing the test-set submission file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    logger = get_logger("run_pipeline")

    try:
        logger.info("Starting House Price Predictor run (project v%s)", config.get(
            "project", "version", default="unknown"
        ))

        training_pipeline = TrainingPipeline(config)
        result = training_pipeline.run()

        summary = {
            "train_metrics": result.train_metrics.to_dict(),
            "holdout_metrics": result.holdout_metrics.to_dict(),
            "cv_rmse_mean": result.trained_model.cv_rmse_mean,
            "cv_rmse_std": result.trained_model.cv_rmse_std,
            "n_features": len(result.trained_model.feature_names),
            "features_used": result.trained_model.feature_names,
        }

        summary_path = config.path("paths", "reports_dir") / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Run summary written to %s", summary_path)

        if not args.skip_prediction:
            prediction_pipeline = PredictionPipeline(config, result.feature_engineer)
            prediction_pipeline.run()

        logger.info("Pipeline run finished successfully.")
        return 0

    except HousePricePredictorError as exc:
        logger.error("Pipeline terminated due to a handled error: %s", exc)
        return 1
    except Exception:  # noqa: BLE001
        logger.exception("Pipeline terminated due to an unhandled error.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
