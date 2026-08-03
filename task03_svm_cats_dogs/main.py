"""Single CLI entry point for the SVM Cats-vs-Dogs pipeline.

Usage:
    python main.py train
    python main.py predict --input data/raw/test1 --output submission.csv
    python main.py generate-data
"""
from __future__ import annotations

import argparse
import sys

from src.data.synthetic_generator import generate_synthetic_dataset
from src.pipeline.prediction_pipeline import run_prediction_pipeline
from src.pipeline.training_pipeline import run_training_pipeline
from src.utils.config_loader import load_config
from src.utils.exceptions import PipelineError
from src.utils.logger import get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SVM Cats vs Dogs pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run the full training pipeline")

    predict_parser = subparsers.add_parser("predict", help="Run inference on a directory of images")
    predict_parser.add_argument("--input", required=True, help="Directory of .jpg images")
    predict_parser.add_argument("--output", default="submission.csv", help="Output CSV filename")

    subparsers.add_parser("generate-data", help="Force (re)generation of the synthetic dataset")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("main")

    try:
        if args.command == "train":
            summary = run_training_pipeline(config_path=args.config)
            logger.info("Training complete. Test accuracy: %.4f", summary["test_metrics"]["accuracy"])

        elif args.command == "predict":
            out_path = run_prediction_pipeline(
                input_dir=args.input, config_path=args.config, output_csv_name=args.output,
            )
            logger.info("Predictions written to %s", out_path)

        elif args.command == "generate-data":
            config = load_config(args.config)
            generate_synthetic_dataset(
                train_dir=config.get("paths", "train_dir"),
                num_images_per_class=config.get("synthetic_data", "num_images_per_class"),
                image_size=config.get("synthetic_data", "image_size"),
                noise_std=config.get("synthetic_data", "noise_std"),
                random_seed=config.get("project", "random_seed"),
            )
        return 0

    except PipelineError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
