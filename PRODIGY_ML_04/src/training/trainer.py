"""Training orchestration: optimizer, scheduler, early stopping, checkpointing."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config_loader import Config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass
class EpochResult:
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


@dataclass
class TrainingHistory:
    epochs: list[EpochResult] = field(default_factory=list)

    def append(self, result: EpochResult) -> None:
        self.epochs.append(result)


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - 1e-5:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(self, cfg: Config, model: nn.Module):
        self.cfg = cfg
        self.device = resolve_device(cfg.training.device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=cfg.training.lr_scheduler_patience,
        )
        self.early_stopper = EarlyStopper(cfg.training.early_stopping_patience)
        self.checkpoint_dir = resolve_path(cfg, cfg.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.checkpoint_dir / cfg.paths.best_model_name
        logger.info("Trainer initialized on device=%s", self.device)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(mode=train)
        total_loss, correct, total = 0.0, 0, 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        history = TrainingHistory()
        best_val_acc = 0.0

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val_loss)

            history.append(EpochResult(train_loss, train_acc, val_loss, val_acc))
            logger.info(
                "Epoch %02d/%02d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch, self.cfg.training.epochs, train_loss, train_acc, val_loss, val_acc,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

            if self.early_stopper.step(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        logger.info("Training complete. Best val_acc=%.4f (checkpoint: %s)",
                     best_val_acc, self.best_path)
        return history

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "val_acc": val_acc,
            "config": {
                "in_channels": self.cfg.model.in_channels,
                "num_classes": self.cfg.model.num_classes,
                "dropout": self.cfg.model.dropout,
                "image_size": list(self.cfg.data.image_size),
                "classes": list(self.cfg.data.classes),
            },
        }, self.best_path)
        logger.debug("Checkpoint saved: %s (val_acc=%.4f)", self.best_path, val_acc)
