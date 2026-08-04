"""Loads images from the folder-per-class raw directory into numpy arrays."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.exceptions import DataLoadError


def load_dataset(cfg: Dict[str, Any], logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    image_size = tuple(cfg["data"]["image_size"])

    if not raw_dir.exists():
        raise DataLoadError(f"Raw data directory does not exist: {raw_dir}")

    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise DataLoadError(f"No class subdirectories found in {raw_dir}")

    images: List[np.ndarray] = []
    labels: List[str] = []
    class_names = [d.name for d in class_dirs]

    for class_dir in class_dirs:
        image_paths = sorted(class_dir.glob("*.jpg"))
        if not image_paths:
            logger.warning("No images found for class '%s'", class_dir.name)
            continue
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize(image_size)
                    images.append(np.asarray(img, dtype=np.uint8))
                    labels.append(class_dir.name)
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable image %s: %s", img_path, exc)

    if not images:
        raise DataLoadError("No valid images were loaded from the raw data directory.")

    X = np.stack(images, axis=0)
    y = np.array(labels)
    logger.info("Loaded %d images across %d classes", len(X), len(class_names))
    return X, y, class_names


def split_dataset(
    X: np.ndarray, y: np.ndarray, cfg: Dict[str, Any], logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_size = cfg["data"]["test_split"]
    random_state = cfg["data"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Split dataset: %d train / %d test", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
