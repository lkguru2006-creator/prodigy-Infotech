"""Synthetic drop-in replacement for the LeapGestRecog Kaggle dataset.

Kaggle dataset layout (https://www.kaggle.com/gti-upm/leapgestrecog):
    leapGestRecog/<subject 00-09>/<gesture class folder>/<frame>.png

This generator reproduces that exact structure with procedurally drawn
grayscale shapes (one distinct geometric signature per gesture class,
perturbed with noise/rotation/translation per sample). Because the
directory layout matches the real dataset byte-for-byte, `GestureDataset`
requires zero changes to consume real Kaggle data later: simply set
`synthetic.enabled: false` and place the extracted archive at
`data.raw_dir`.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.config_loader import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Native resolution similar in spirit to the source IR camera frames;
# images are downsized later by the preprocessing transform.
_CANVAS_SIZE = (160, 120)


def _draw_gesture(class_idx: int, canvas_size: tuple[int, int], rng: random.Random,
                   noise_std: float) -> Image.Image:
    """Render one procedural grayscale image with a class-specific shape."""
    w, h = canvas_size
    img = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(img)

    cx = w // 2 + rng.randint(-10, 10)
    cy = h // 2 + rng.randint(-8, 8)
    scale = rng.uniform(0.85, 1.15)
    fill = rng.randint(200, 255)

    # Each class gets a deterministic, visually distinct base geometry so
    # the classification task is learnable but not trivial (noise + jitter
    # added afterwards).
    if class_idx == 0:      # 01_palm - open circle
        r = int(28 * scale)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif class_idx == 1:    # 02_l - L shape
        draw.rectangle([cx - 25, cy - 30, cx - 10, cy + 30], fill=fill)
        draw.rectangle([cx - 25, cy + 15, cx + 25, cy + 30], fill=fill)
    elif class_idx == 2:    # 03_fist - filled square
        r = int(22 * scale)
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif class_idx == 3:    # 04_fist_moved - square offset from center
        r = int(20 * scale)
        ox, oy = rng.randint(15, 30), rng.randint(-10, 10)
        draw.rectangle([cx - r + ox, cy - r + oy, cx + r + ox, cy + r + oy], fill=fill)
    elif class_idx == 4:    # 05_thumb - narrow vertical ellipse
        draw.ellipse([cx - 10, cy - 32, cx + 10, cy + 32], fill=fill)
    elif class_idx == 5:    # 06_index - thin vertical bar
        draw.rectangle([cx - 6, cy - 35, cx + 6, cy + 35], fill=fill)
    elif class_idx == 6:    # 07_ok - ring (circle with hole)
        r = int(28 * scale)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        r2 = int(r * 0.5)
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=0)
    elif class_idx == 7:    # 08_palm_moved - circle offset
        r = int(26 * scale)
        ox, oy = rng.randint(-25, 25), rng.randint(-20, 20)
        draw.ellipse([cx - r + ox, cy - r + oy, cx + r + ox, cy + r + oy], fill=fill)
    elif class_idx == 8:    # 09_c - open arc (letter C)
        r = int(30 * scale)
        draw.arc([cx - r, cy - r, cx + r, cy + r], start=40, end=320, fill=fill, width=10)
    else:                   # 10_down - downward triangle
        pts = [(cx, cy + 30), (cx - 28, cy - 25), (cx + 28, cy - 25)]
        draw.polygon(pts, fill=fill)

    # small random rotation for intra-class variety
    angle = rng.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, noise_std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def generate_synthetic_dataset(cfg: Config, force: bool = False) -> Path:
    """Generate the full synthetic dataset tree under cfg.data.raw_dir.

    Returns the path to the generated dataset root. Idempotent: if the
    directory already contains data and `force` is False, generation is
    skipped.
    """
    raw_dir = resolve_path(cfg, cfg.data.raw_dir)
    if raw_dir.exists() and any(raw_dir.iterdir()) and not force:
        logger.info("Synthetic dataset already present at %s (skipping generation)", raw_dir)
        return raw_dir

    rng = random.Random(cfg.project.seed)
    np.random.seed(cfg.project.seed)

    classes: list[str] = cfg.data.classes
    n_subjects: int = cfg.data.num_subjects
    n_per_class: int = cfg.data.images_per_class_per_subject
    noise_std: float = cfg.synthetic.noise_std

    total = 0
    for subject_idx in range(n_subjects):
        subject_name = f"{subject_idx:02d}"
        for class_idx, class_name in enumerate(classes):
            class_dir = raw_dir / subject_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                img = _draw_gesture(class_idx, _CANVAS_SIZE, rng, noise_std)
                fname = f"frame_{subject_name}_{class_idx:02d}_{i:04d}.png"
                img.save(class_dir / fname)
                total += 1

    logger.info(
        "Generated synthetic dataset: %d subjects x %d classes x %d images = %d files at %s",
        n_subjects, len(classes), n_per_class, total, raw_dir,
    )
    return raw_dir
