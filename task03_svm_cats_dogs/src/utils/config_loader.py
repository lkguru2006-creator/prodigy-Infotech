"""Configuration loading with project-root-anchored absolute paths.

Lesson learned (Task-01 / Task-02): resolving relative paths against the
process current-working-directory scatters output artifacts whenever the
pipeline is invoked from outside the project root. Every path in the
config is therefore rewritten here to an absolute path anchored at the
project root, once, at load time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.exceptions import ConfigError

# project root = two levels above this file (src/utils/config_loader.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PATH_KEYS = (
    "data_raw_dir", "train_dir", "test_dir", "processed_dir",
    "model_dir", "metrics_dir", "figures_dir", "logs_dir", "outputs_dir",
)


class AppConfig:
    """Thin, attribute-friendly wrapper around the parsed YAML config."""

    def __init__(self, raw: dict[str, Any]):
        self._raw = raw

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self._raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """Load the YAML config and resolve every path entry to an absolute path."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise ConfigError(f"Config file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {exc}") from exc

    if not raw or "paths" not in raw:
        raise ConfigError("Config is missing required 'paths' section.")

    for key in _PATH_KEYS:
        if key in raw["paths"]:
            resolved = (PROJECT_ROOT / raw["paths"][key]).resolve()
            raw["paths"][key] = str(resolved)

    return AppConfig(raw)


def ensure_output_dirs(config: AppConfig) -> None:
    """Create every directory referenced under paths.* if it does not exist."""
    for key in _PATH_KEYS:
        dir_path = config.get("paths", key)
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
