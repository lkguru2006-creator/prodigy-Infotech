"""Tests for src.utils.config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config_loader import load_config
from src.utils.exceptions import ConfigurationError


def test_load_config_success(tmp_path: Path, sample_config: dict):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(yaml.dump(sample_config))

    loaded = load_config(str(config_file))

    assert loaded["project"]["name"] == "test-project"
    assert loaded["model"]["n_clusters"] == 5


def test_load_config_missing_file_raises(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(str(missing_path))


def test_load_config_missing_required_section_raises(tmp_path: Path, sample_config: dict):
    del sample_config["model"]
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(yaml.dump(sample_config))

    with pytest.raises(ConfigurationError, match="missing required"):
        load_config(str(config_file))


def test_load_config_malformed_yaml_raises(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text("project: [unterminated\n  bad: yaml: : :")

    with pytest.raises(ConfigurationError):
        load_config(str(config_file))


def test_load_config_resolves_relative_paths_to_project_root(tmp_path: Path, sample_config: dict):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(yaml.dump(sample_config))

    loaded = load_config(str(config_file))

    # raw_path should now be absolute and rooted at tmp_path, not at cwd
    assert Path(loaded["data"]["raw_path"]).is_absolute()
    assert str(tmp_path) in loaded["data"]["raw_path"]
    assert Path(loaded["paths"]["model_dir"]).is_absolute()
