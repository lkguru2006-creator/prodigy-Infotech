"""Generate (or regenerate) the synthetic LeapGestRecog-structured dataset.

Usage:
    python scripts/generate_data.py [--force] [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config, DEFAULT_CONFIG_PATH
from src.data.synthetic_generator import generate_synthetic_dataset
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic gesture dataset")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--force", action="store_true", help="Regenerate even if data exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, str(cfg.paths.log_dir), cfg.logging.log_to_file)
    logger = get_logger(__name__)
    set_global_seed(cfg.project.seed)

    if not cfg.synthetic.enabled:
        logger.warning("synthetic.enabled is False in config; skipping generation. "
                        "Set it to true, or place real data manually at %s", cfg.data.raw_dir)
        return

    generate_synthetic_dataset(cfg, force=args.force)


if __name__ == "__main__":
    main()
