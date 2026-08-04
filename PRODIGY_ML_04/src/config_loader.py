"""Loads and validates config/config.yaml into a simple typed structure.

Using a dataclass-like namespace (instead of raw dict indexing everywhere)
keeps the rest of the codebase free of magic string lookups and typos.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


class Config(SimpleNamespace):
    """Namespace wrapper with a couple of convenience helpers."""

    @property
    def root_dir(self) -> Path:
        return DEFAULT_CONFIG_PATH.parent.parent


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ns = _to_namespace(raw)
    cfg = Config(**vars(ns))
    return cfg


def resolve_path(cfg: Config, relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (cfg.root_dir / p)
