"""
Centralized logging configuration.

Every module in this project obtains its logger via `get_logger(__name__)`
rather than using `print`. This gives consistent, leveled, timestamped
output and a persistent rotating log file under logs/.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from house_price_predictor.utils.config import AppConfig

_CONFIGURED = False


def setup_logging(config: AppConfig) -> None:
    """Configure the root logger once, based on config.yaml settings."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = config.get("logging", "level", default="INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)

    log_dir = config.path("logging", "log_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / config.get("logging", "log_file", default="pipeline.log")

    max_bytes = int(config.get("logging", "max_bytes", default=5_242_880))
    backup_count = int(config.get("logging", "backup_count", default=3))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Call setup_logging() first."""
    return logging.getLogger(name)
