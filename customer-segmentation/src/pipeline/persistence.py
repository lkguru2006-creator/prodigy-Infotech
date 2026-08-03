"""
Persistence utilities for model artifacts: trained model, scaler, and
associated metadata. Centralizing save/load logic here ensures consistent
error handling and avoids scattering joblib/json calls across the codebase.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

from src.utils.exceptions import ModelPersistenceError


def save_artifact(obj: Any, path: str | Path, logger: logging.Logger | None = None) -> None:
    """Persist a Python object (model, scaler, etc.) via joblib."""
    out_path = Path(path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, out_path)
    except Exception as exc:
        raise ModelPersistenceError(f"Failed to save artifact to '{out_path}': {exc}") from exc
    if logger:
        logger.info("Saved artifact to '%s'.", out_path)


def load_artifact(path: str | Path, logger: logging.Logger | None = None) -> Any:
    """Load a previously persisted joblib artifact."""
    in_path = Path(path)
    if not in_path.exists():
        raise ModelPersistenceError(f"Artifact not found at '{in_path}'.")
    try:
        obj = joblib.load(in_path)
    except Exception as exc:
        raise ModelPersistenceError(f"Failed to load artifact from '{in_path}': {exc}") from exc
    if logger:
        logger.info("Loaded artifact from '%s'.", in_path)
    return obj


def save_json(data: dict[str, Any], path: str | Path, logger: logging.Logger | None = None) -> None:
    """Persist a dictionary as formatted JSON."""
    out_path = Path(path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        raise ModelPersistenceError(f"Failed to write JSON to '{out_path}': {exc}") from exc
    if logger:
        logger.info("Saved JSON artifact to '%s'.", out_path)
