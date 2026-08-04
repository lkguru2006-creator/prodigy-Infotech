"""Generates synthetic food images matching the Food-101 folder-per-class schema:

    data/raw/<class_name>/<class_name>_<idx>.jpg

Each class gets a distinct, deterministic color/texture signature (seeded by class
name) so the classifier has real, learnable signal instead of pure noise. Once real
Food-101 images are placed in data/raw/<class_name>/ following the same schema,
set synthetic.enabled: false in config.yaml and no code changes are required.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from src.utils.exceptions import DataGenerationError


def _seed_for_class(class_name: str) -> int:
    return int(hashlib.sha256(class_name.encode("utf-8")).hexdigest(), 16) % (2**32)


def _synthesize_image(class_name: str, size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = size
    base_rng = np.random.default_rng(_seed_for_class(class_name))
    base_color = base_rng.integers(40, 216, size=3)

    # Base canvas with per-class base color + per-image gaussian texture noise
    canvas = np.tile(base_color, (h, w, 1)).astype(np.float32)
    texture = rng.normal(loc=0.0, scale=18.0, size=(h, w, 3))
    canvas += texture

    # Add a class-specific radial pattern so classes are visually separable
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    freq = 0.15 + (_seed_for_class(class_name) % 10) * 0.03
    ring = (np.sin(radius * freq) * 25).astype(np.float32)
    canvas += ring[..., None]

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas


def generate_synthetic_dataset(cfg: Dict[str, Any], logger: logging.Logger) -> None:
    syn_cfg = cfg["synthetic"]
    raw_dir = Path(cfg["data"]["raw_dir"])
    image_size = tuple(cfg["data"]["image_size"])
    class_names = syn_cfg["class_names"][: syn_cfg["num_classes"]]
    images_per_class = syn_cfg["images_per_class"]
    rng = np.random.default_rng(cfg["data"]["random_state"])

    try:
        for class_name in class_names:
            class_dir = raw_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            existing = list(class_dir.glob("*.jpg"))
            if len(existing) >= images_per_class:
                logger.info("Skipping %s: %d images already present", class_name, len(existing))
                continue
            for idx in range(images_per_class):
                img_array = _synthesize_image(class_name, image_size, rng)
                img = Image.fromarray(img_array, mode="RGB")
                img.save(class_dir / f"{class_name}_{idx:04d}.jpg", quality=90)
            logger.info("Generated %d synthetic images for class '%s'", images_per_class, class_name)
    except OSError as exc:
        raise DataGenerationError(f"Failed to write synthetic images: {exc}") from exc

    logger.info("Synthetic dataset generation complete: %d classes at %s", len(class_names), raw_dir)
