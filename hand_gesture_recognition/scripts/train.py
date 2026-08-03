"""Train the gesture recognition model end-to-end.

Usage:
    python scripts/train.py [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.config_loader import load_config, resolve_path, DEFAULT_CONFIG_PATH
from src.data.dataset import (GestureDataset, build_transforms, index_dataset,
                               stratified_split)
from src.data.synthetic_generator import generate_synthetic_dataset
from src.models.cnn_model import build_model
from src.training.trainer import Trainer
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hand gesture recognition model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, str(resolve_path(cfg, cfg.paths.log_dir)),
                       cfg.logging.log_to_file)
    logger = get_logger(__name__)
    set_global_seed(cfg.project.seed)

    raw_dir = resolve_path(cfg, cfg.data.raw_dir)
    if cfg.synthetic.enabled and not raw_dir.exists():
        logger.info("Raw data not found; generating synthetic dataset first.")
        generate_synthetic_dataset(cfg)

    samples, class_to_idx = index_dataset(cfg)
    train_s, val_s, test_s = stratified_split(
        samples, cfg.data.val_split, cfg.data.test_split, cfg.project.seed
    )

    train_ds = GestureDataset(train_s, build_transforms(cfg, train=True))
    val_ds = GestureDataset(val_s, build_transforms(cfg, train=False))

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                             shuffle=False, num_workers=0)

    model = build_model(cfg)
    trainer = Trainer(cfg, model)
    trainer.fit(train_loader, val_loader)

    logger.info("Training pipeline finished. Best checkpoint: %s", trainer.best_path)


if __name__ == "__main__":
    main()
