"""Centralized logging configuration.

Every module in this project imports `get_logger` instead of using print().
This guarantees consistent, timestamped, leveled output and makes it trivial
to redirect logs to files/monitoring systems in production.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def configure_logging(level: str = "INFO", log_dir: str | None = None,
                       log_to_file: bool = True) -> None:
    """Configure the root logger once for the whole process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_to_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "pipeline.log")
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Configures defaults if not yet set."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
