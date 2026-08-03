"""Single entry point for the food-recognition + calorie-estimation pipeline.

Usage:
    python main.py [--config config/config.yaml]
"""
from __future__ import annotations

import argparse
import sys

from src.pipeline.pipeline import run_pipeline
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Food recognition + calorie estimation pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - config errors must not crash with a raw traceback
        print(f"FATAL: failed to load config: {exc}", file=sys.stderr)
        return 1

    logger = get_logger("food_calorie_pipeline", cfg)

    try:
        metrics = run_pipeline(cfg, logger)
    except Exception:
        logger.error("Pipeline execution failed. See log for details.")
        return 1

    logger.info("Final accuracy: %.4f | Final F1 (macro): %.4f", metrics["accuracy"], metrics["f1_macro"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
