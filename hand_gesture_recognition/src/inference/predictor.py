"""Production inference wrapper: load checkpoint once, predict on demand."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.cnn_model import GestureCNN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GesturePredictor:
    """Self-contained predictor: reconstructs model + preprocessing from
    metadata embedded in the checkpoint, so no external config is required
    at inference time.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else
            ("cpu" if device == "auto" else device)
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt["config"]

        self.classes: list[str] = cfg["classes"]
        self.model = GestureCNN(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(tuple(cfg["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        logger.info("Loaded predictor from %s (val_acc=%.4f, %d classes)",
                     checkpoint_path, ckpt.get("val_acc", -1), len(self.classes))

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        with Image.open(image_path) as img:
            tensor = self.transform(img.convert("L")).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        top_idx = int(torch.argmax(probs).item())

        return {
            "predicted_class": self.classes[top_idx],
            "confidence": float(probs[top_idx]),
            "class_probabilities": {
                self.classes[i]: float(probs[i]) for i in range(len(self.classes))
            },
        }
