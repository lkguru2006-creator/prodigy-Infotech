"""Structured rotating logger factory. No raw print statements anywhere in the pipeline."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str, cfg: Dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls

    level = getattr(logging, cfg["logging"].get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    log_dir = Path(cfg["outputs"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=cfg["logging"].get("max_bytes", 1_048_576),
        backupCount=cfg["logging"].get("backup_count", 3),
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
