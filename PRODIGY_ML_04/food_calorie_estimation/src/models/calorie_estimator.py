"""Maps a predicted food class to an estimated calorie value (kcal/100g).

Lookup-table approach: simple, deterministic, and auditable — appropriate given
the classifier's output is a discrete food class. Swap for a regression head
(trained on portion-annotated data) if per-image calorie regression is required.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.utils.exceptions import CalorieEstimationError


class CalorieEstimator:
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.lookup: Dict[str, int] = cfg["calorie_lookup"]
        self.default = self.lookup.get("default", 250)
        self.logger = logger

    def estimate(self, class_name: str) -> int:
        if class_name not in self.lookup:
            self.logger.warning("No calorie entry for class '%s'; using default", class_name)
            return self.default
        return self.lookup[class_name]

    def estimate_batch(self, class_names: List[str]) -> List[int]:
        try:
            return [self.estimate(c) for c in class_names]
        except Exception as exc:  # noqa: BLE001
            raise CalorieEstimationError(f"Batch calorie estimation failed: {exc}") from exc
