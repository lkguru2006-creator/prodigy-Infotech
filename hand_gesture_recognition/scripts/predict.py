"""Predict the gesture class for a single image.

Usage:
    python scripts/predict.py --image path/to/image.png [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config, resolve_path, DEFAULT_CONFIG_PATH
from src.inference.predictor import GesturePredictor
from src.utils.logger import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single gesture image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, str(resolve_path(cfg, cfg.paths.log_dir)),
                       cfg.logging.log_to_file)
    logger = get_logger(__name__)

    checkpoint_path = resolve_path(cfg, cfg.paths.checkpoint_dir) / cfg.paths.best_model_name
    predictor = GesturePredictor(checkpoint_path, device=cfg.training.device)
    result = predictor.predict(args.image)

    logger.info("Prediction: %s (confidence=%.4f)",
                result["predicted_class"], result["confidence"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
