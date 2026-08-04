"""Evaluate the best checkpoint on the held-out test split.

Usage:
    python scripts/evaluate.py [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config_loader import load_config, resolve_path, DEFAULT_CONFIG_PATH
from src.data.dataset import (GestureDataset, build_transforms, index_dataset,
                               stratified_split)
from src.evaluation.evaluator import evaluate_model, save_report
from src.models.cnn_model import build_model
from src.training.trainer import resolve_device
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hand gesture recognition model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, str(resolve_path(cfg, cfg.paths.log_dir)),
                       cfg.logging.log_to_file)
    logger = get_logger(__name__)
    set_global_seed(cfg.project.seed)

    checkpoint_path = resolve_path(cfg, cfg.paths.checkpoint_dir) / cfg.paths.best_model_name
    if not checkpoint_path.exists():
        logger.error("No checkpoint found at %s. Run scripts/train.py first.", checkpoint_path)
        return

    samples, _ = index_dataset(cfg)
    _, _, test_s = stratified_split(samples, cfg.data.val_split, cfg.data.test_split,
                                     cfg.project.seed)
    test_ds = GestureDataset(test_s, build_transforms(cfg, train=False))
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    device = resolve_device(cfg.training.device)
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    result = evaluate_model(model, test_loader, device, cfg.data.classes)
    save_report(cfg, result)


if __name__ == "__main__":
    main()
