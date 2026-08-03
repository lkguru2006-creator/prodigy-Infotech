"""Loads YAML config and resolves all paths to absolute, project-root-anchored paths.

This exists specifically to avoid the recurring bug where relative paths resolve
against the current working directory (wherever the script was invoked from)
instead of the project root, which scatters output artifacts.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.exceptions import ConfigError

# Project root = two levels up from this file (src/utils/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PATH_KEYS = {
    ("data", "raw_dir"),
    ("data", "processed_dir"),
    ("outputs", "models_dir"),
    ("outputs", "metrics_dir"),
    ("outputs", "logs_dir"),
    ("outputs", "predictions_dir"),
}


def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(cfg)
    for section, key in _PATH_KEYS:
        try:
            raw_value = resolved[section][key]
        except KeyError as exc:
            raise ConfigError(f"Missing required config key: {section}.{key}") from exc
        abs_path = (PROJECT_ROOT / raw_value).resolve()
        resolved[section][key] = str(abs_path)
    resolved["project"]["root"] = str(PROJECT_ROOT)
    return resolved


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load config.yaml and return a dict with all paths made absolute."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {exc}") from exc

    if not raw_cfg:
        raise ConfigError("Config file is empty or invalid.")

    return _resolve_paths(raw_cfg)


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    """Create all output directories declared in config if they don't exist."""
    for key in ("models_dir", "metrics_dir", "logs_dir", "predictions_dir"):
        Path(cfg["outputs"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
