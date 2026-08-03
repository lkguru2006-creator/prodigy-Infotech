"""Model evaluation: confusion matrix, per-class metrics, JSON report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.config_loader import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device,
                    class_names: list[str]) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    result = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1_score": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
        "confusion_matrix": cm,
    }
    return result


def save_report(cfg: Config, result: dict, filename: str = "evaluation_report.json") -> Path:
    report_dir = resolve_path(cfg, cfg.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Evaluation report saved to %s (accuracy=%.4f, macro_f1=%.4f)",
                out_path, result["accuracy"], result["macro_f1"])
    return out_path
