"""Synthetic drop-in stand-in for the Kaggle Dogs vs Cats dataset.

Real data usage
----------------
Download https://www.kaggle.com/c/dogs-vs-cats/data, unzip ``train.zip``
and drop the resulting ``cat.N.jpg`` / ``dog.N.jpg`` files directly into
``data/raw/train/``. No code changes are required -- the loader and every
downstream stage only depend on the Kaggle filename convention, not on
how the files were produced.

Synthetic strategy
-------------------
Real photographs of cats/dogs are not learnable from statistical
first principles, so this generator instead manufactures images whose
low-level statistics (dominant hue, texture frequency, edge density)
differ *by class* -- enough for HOG + color-histogram features to expose
a learnable decision boundary, which is exactly what the downstream
feature extractor operates on. This keeps the full pipeline exercised
end-to-end without requiring the real dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.exceptions import DataError
from src.utils.logger import get_logger

logger = get_logger("data.synthetic_generator")


def _synthetic_cat_image(size: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Warm-toned, higher-frequency texture stand-in for a cat photo."""
    base = rng.normal(loc=[150, 120, 95], scale=noise_std, size=(size, size, 3))
    xx, yy = np.meshgrid(np.linspace(0, 6 * np.pi, size), np.linspace(0, 6 * np.pi, size))
    texture = (np.sin(xx) * np.cos(yy))[..., None] * 25
    img = base + texture
    return np.clip(img, 0, 255).astype(np.uint8)


def _synthetic_dog_image(size: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Cool-toned, lower-frequency blobby texture stand-in for a dog photo."""
    base = rng.normal(loc=[110, 130, 150], scale=noise_std, size=(size, size, 3))
    xx, yy = np.meshgrid(np.linspace(0, 2 * np.pi, size), np.linspace(0, 2 * np.pi, size))
    texture = (np.sin(xx * 1.5) + np.cos(yy * 1.5))[..., None] * 20
    img = base + texture
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic_dataset(
    train_dir: str | Path,
    num_images_per_class: int,
    image_size: int,
    noise_std: float,
    random_seed: int = 42,
) -> int:
    """Populate ``train_dir`` with synthetic cat.N.jpg / dog.N.jpg files.

    Returns the total number of images written. No-op (with a log line)
    if the directory already contains files matching the Kaggle
    convention, so real data is never silently overwritten.
    """
    train_path = Path(train_dir)
    train_path.mkdir(parents=True, exist_ok=True)

    existing = list(train_path.glob("*.jpg"))
    if existing:
        logger.info("Found %d existing image(s) in %s; skipping synthetic generation.",
                     len(existing), train_path)
        return len(existing)

    rng = np.random.default_rng(random_seed)
    written = 0
    try:
        for idx in range(num_images_per_class):
            cat_img = Image.fromarray(_synthetic_cat_image(image_size, noise_std, rng))
            cat_img.save(train_path / f"cat.{idx}.jpg", quality=90)

            dog_img = Image.fromarray(_synthetic_dog_image(image_size, noise_std, rng))
            dog_img.save(train_path / f"dog.{idx}.jpg", quality=90)
            written += 2
    except OSError as exc:
        raise DataError(f"Failed writing synthetic images to {train_path}: {exc}") from exc

    logger.info("Generated %d synthetic images (%d per class) in %s",
                written, num_images_per_class, train_path)
    return written
