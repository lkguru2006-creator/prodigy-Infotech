"""
Configuration loading and validation utilities.

Loads config/config.yaml once and validates its structure so failures
surface immediately and clearly, rather than as a KeyError three modules
deep into the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.exceptions import ConfigurationError

_REQUIRED_TOP_LEVEL_KEYS = (
    "project",
    "data",
    "features",
    "model",
    "evaluation",
    "paths",
    "logging",
)


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """
    Load and validate the pipeline configuration from a YAML file.

    All relative paths declared under the config's `data` and `paths`
    sections are resolved against the project root (the parent of the
    `config/` folder containing this file) and rewritten as absolute
    paths. This makes pipeline behavior independent of the caller's
    current working directory -- running the entry-point script from the
    project root or from anywhere else on disk produces identical
    results, instead of silently writing artifacts into whatever
    directory happened to be the current working directory.

    Raises:
        ConfigurationError: if the file is missing, malformed, or missing
            required sections.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found at '{path}'. "
            "Ensure config/config.yaml exists relative to the project root."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML config at '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigurationError(f"Config at '{path}' did not parse into a dictionary.")

    missing = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ConfigurationError(
            f"Config at '{path}' is missing required top-level section(s): {missing}"
        )

    # Project root is the parent of the 'config' folder containing this file.
    project_root = path.parent.parent if path.parent.name == "config" else path.parent
    _resolve_paths_in_place(config.get("data", {}), project_root, keys=("raw_path", "processed_path"))
    _resolve_paths_in_place(
        config.get("paths", {}),
        project_root,
        keys=("model_dir", "metrics_dir", "plots_dir", "log_dir"),
    )

    return config


def _resolve_paths_in_place(section: dict[str, Any], project_root: Path, keys: tuple[str, ...]) -> None:
    """Rewrite specified relative-path config values to absolute paths in place."""
    for key in keys:
        if key in section and section[key] is not None:
            candidate = Path(section[key])
            if not candidate.is_absolute():
                section[key] = str((project_root / candidate).resolve())
