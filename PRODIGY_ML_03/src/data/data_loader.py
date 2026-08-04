"""Discovery and loading of Kaggle-convention cat/dog image files.

Kaggle filename convention: ``{label}.{index}.jpg`` inside a single
train directory (e.g. ``cat.137.jpg``), and unlabeled ``{id}.jpg`` files
inside a test directory. This module only depends on that convention,
so a real Kaggle download is a pure drop-in replacement for the
synthetic generator's output.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.exceptions import DataError
from src.utils.logger import get_logger

logger = get_logger("data.data_loader")

LABEL_MAP = {"cat": 0, "dog": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int


def discover_train_records(train_dir: str | Path) -> list[ImageRecord]:
    """Scan train_dir and parse (path, label) pairs from filenames."""
    train_path = Path(train_dir)
    if not train_path.exists():
        raise DataError(f"Train directory does not exist: {train_path}")

    records: list[ImageRecord] = []
    for file_path in sorted(train_path.glob("*.jpg")):
        stem_parts = file_path.stem.split(".")
        label_str = stem_parts[0].lower()
        if label_str not in LABEL_MAP:
            logger.warning("Skipping file with unrecognized label prefix: %s", file_path.name)
            continue
        records.append(ImageRecord(path=file_path, label=LABEL_MAP[label_str]))

    if not records:
        raise DataError(f"No valid cat/dog images found under {train_path}")

    logger.info("Discovered %d labeled images under %s", len(records), train_path)
    return records


def load_image_as_array(path: Path, resize_dim: int, color_mode: str = "RGB") -> np.ndarray:
    """Load a single image, resize, and return as a uint8 HxWxC array."""
    try:
        with Image.open(path) as img:
            img = img.convert(color_mode)
            img = img.resize((resize_dim, resize_dim), Image.BILINEAR)
            return np.array(img, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise DataError(f"Failed to load image {path}: {exc}") from exc


def load_dataset_arrays(
    records: list[ImageRecord], resize_dim: int, color_mode: str = "RGB"
) -> tuple[np.ndarray, np.ndarray]:
    """Load every record into a stacked image array and label array."""
    images = np.stack([load_image_as_array(r.path, resize_dim, color_mode) for r in records])
    labels = np.array([r.label for r in records], dtype=np.int64)
    return images, labels


def train_val_test_split(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    val_size: float,
    stratify: bool,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Stratified split into train / val / test partitions."""
    strat = labels if stratify else None
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_seed, stratify=strat
    )

    strat2 = y_train_full if stratify else None
    relative_val = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=relative_val,
        random_state=random_seed, stratify=strat2,
    )

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(x_train), len(x_val), len(x_test),
    )
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }
