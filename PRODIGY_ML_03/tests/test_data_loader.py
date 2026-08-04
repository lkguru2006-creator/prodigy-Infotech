"""Pytest-syntax tests for the data loader / synthetic generator."""
import numpy as np
import pytest

from src.data.data_loader import discover_train_records, train_val_test_split
from src.utils.exceptions import DataError


def test_discover_train_records_missing_dir(tmp_path):
    with pytest.raises(DataError):
        discover_train_records(tmp_path / "does_not_exist")


def test_discover_train_records_parses_labels(tmp_path):
    (tmp_path / "cat.0.jpg").write_bytes(b"fake")
    (tmp_path / "dog.0.jpg").write_bytes(b"fake")
    records = discover_train_records(tmp_path)
    labels = sorted(r.label for r in records)
    assert labels == [0, 1]


def test_train_val_test_split_sizes():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(100, 8, 8, 3), dtype=np.uint8)
    labels = np.array([0, 1] * 50)
    splits = train_val_test_split(images, labels, test_size=0.2, val_size=0.1,
                                    stratify=True, random_seed=42)
    assert len(splits["x_train"]) + len(splits["x_val"]) + len(splits["x_test"]) == 100
