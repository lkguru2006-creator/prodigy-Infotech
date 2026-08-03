"""
Configuration loader.

Loads the project's YAML configuration into a single, validated object so
the rest of the codebase never has to deal with raw dict lookups or
hardcoded paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


@dataclass(frozen=True)
class AppConfig:
    """Typed, read-only wrapper around the raw YAML configuration."""

    raw: dict[str, Any] = field(repr=False)
    project_root: Path

    # ---- convenience accessors -------------------------------------------------

    def get(self, *keys: str, default: Any = None) -> Any:
        """Safely walk a nested key path, e.g. cfg.get('model', 'type')."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, *keys: str) -> Path:
        """Resolve a configured relative path to an absolute Path."""
        rel = self.get(*keys)
        if rel is None:
            raise ConfigError(f"Missing path config for key path: {keys}")
        return self.project_root / rel

    @property
    def random_seed(self) -> int:
        return int(self.get("project", "random_seed", default=42))

    @property
    def target_column(self) -> str:
        return self.get("data", "target_column", default="SalePrice")

    @property
    def id_column(self) -> str:
        return self.get("data", "id_column", default="Id")


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load configuration from YAML.

    Parameters
    ----------
    config_path:
        Optional explicit path to a config.yaml. If not provided, resolves
        relative to the project root (two levels above this file).

    Returns
    -------
    AppConfig
    """
    here = Path(__file__).resolve()
    project_root = here.parents[3]  # src/house_price_predictor/utils -> root

    if config_path is None:
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError("Config file is empty or malformed.")

    return AppConfig(raw=raw, project_root=project_root)
