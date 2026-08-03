"""
Structured logging configuration with file rotation.

No raw print statements are used anywhere in this pipeline. All runtime
information flows through this logger, which writes to both console and
a rotating log file for observability across runs.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(
    name: str,
    log_dir: str | Path = "outputs/logs",
    log_filename: str = "pipeline.log",
    level: str = "INFO",
    max_bytes: int = 1_048_576,
    backup_count: int = 3,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> logging.Logger:
    """
    Create or retrieve a configured logger that writes to console and a
    rotating log file. Safe to call multiple times with the same name;
    handlers are only attached once per logger.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir_path / log_filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
