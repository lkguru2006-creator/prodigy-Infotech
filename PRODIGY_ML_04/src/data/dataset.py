"""Dataset loading for LeapGestRecog-structured data (real or synthetic).

Scans: raw_dir/<subject>/<class_folder>/<image>.png
Works identically whether raw_dir holds real Kaggle files or synthetic
files produced by `synthetic_generator.py`, since both share the same
directory contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.config_loader import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class Sample:
    path: Path
    label: int


def index_dataset(cfg: Config) -> tuple[list[Sample], dict[str, int]]:
    """Walk raw_dir and build (path, label) pairs plus a class->idx map."""
    raw_dir = resolve_path(cfg, cfg.data.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. "
            "Run scripts/generate_data.py or place the extracted Kaggle "
            "dataset at this path."
        )

    class_names: list[str] = cfg.data.classes
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples: list[Sample] = []
    for subject_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            if class_dir.name not in class_to_idx:
                continue
            label = class_to_idx[class_dir.name]
            for f in class_dir.iterdir():
                if f.suffix.lower() in VALID_EXTENSIONS:
                    samples.append(Sample(path=f, label=label))

    if not samples:
        raise RuntimeError(f"No images found under {raw_dir}")

    logger.info("Indexed %d images across %d classes", len(samples), len(class_names))
    return samples, class_to_idx


def stratified_split(samples: list[Sample], val_split: float, test_split: float,
                      seed: int) -> tuple[list[Sample], list[Sample], list[Sample]]:
    labels = [s.label for s in samples]
    train_val, test = train_test_split(
        samples, test_size=test_split, stratify=labels, random_state=seed
    )
    val_ratio_of_remainder = val_split / (1.0 - test_split)
    train_labels = [s.label for s in train_val]
    train, val = train_test_split(
        train_val, test_size=val_ratio_of_remainder, stratify=train_labels, random_state=seed
    )
    logger.info("Split sizes -> train: %d, val: %d, test: %d", len(train), len(val), len(test))
    return train, val, test


def build_transforms(cfg: Config, train: bool) -> transforms.Compose:
    size = tuple(cfg.data.image_size)
    ops: list = [transforms.Grayscale(num_output_channels=1), transforms.Resize(size)]
    if train:
        ops += [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    return transforms.Compose(ops)


class GestureDataset(Dataset):
    """Loads images lazily from disk to keep memory usage bounded."""

    def __init__(self, samples: list[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            img = img.convert("L")
            tensor = self.transform(img)
        return tensor, sample.label
