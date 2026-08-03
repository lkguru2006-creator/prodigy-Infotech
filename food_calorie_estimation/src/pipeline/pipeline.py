"""Orchestrates the full food-classification + calorie-estimation pipeline."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.data.data_loader import load_dataset, split_dataset
from src.data.synthetic_generator import generate_synthetic_dataset
from src.features.feature_extractor import FoodFeatureExtractor
from src.models.calorie_estimator import CalorieEstimator
from src.models.sklearn_classifier import SklearnFoodClassifier
from src.utils.config_loader import ensure_output_dirs
from src.utils.exceptions import FoodCalorieError


class FoodCalorPipeline:
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        ensure_output_dirs(cfg)
        self.feature_extractor = FoodFeatureExtractor(cfg, logger)
        self.classifier = SklearnFoodClassifier(cfg, logger)
        self.calorie_estimator = CalorieEstimator(cfg, logger)

    def run(self) -> Dict[str, Any]:
        run_start = time.time()
        self.logger.info("Pipeline run started")

        if self.cfg["synthetic"]["enabled"]:
            generate_synthetic_dataset(self.cfg, self.logger)

        X, y, class_names = load_dataset(self.cfg, self.logger)
        X_train, X_test, y_train, y_test = split_dataset(X, y, self.cfg, self.logger)

        X_train_feat = self.feature_extractor.fit_transform(X_train)
        X_test_feat = self.feature_extractor.transform(X_test)

        self.classifier.fit(X_train_feat, y_train)
        y_pred = self.classifier.predict(X_test_feat)

        metrics = self._compute_metrics(y_test, y_pred)
        self._save_artifacts(metrics, y_test, y_pred, class_names, run_start)

        self.logger.info("Pipeline run complete in %.2fs | accuracy=%.4f",
                          time.time() - run_start, metrics["accuracy"])
        return metrics

    def _compute_metrics(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

    def _save_artifacts(
        self,
        metrics: Dict[str, Any],
        y_test: np.ndarray,
        y_pred: np.ndarray,
        class_names: list,
        run_start: float,
    ) -> None:
        models_dir = Path(self.cfg["outputs"]["models_dir"])
        metrics_dir = Path(self.cfg["outputs"]["metrics_dir"])
        predictions_dir = Path(self.cfg["outputs"]["predictions_dir"])

        self.classifier.save(models_dir / "food_classifier.joblib")

        with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        summary = {
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - run_start, 2),
            "model_backend": self.cfg["model"]["backend"],
            "num_classes": len(class_names),
            "test_samples": len(y_test),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        }
        with open(metrics_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        pred_calories = self.calorie_estimator.estimate_batch(list(y_pred))
        true_calories = self.calorie_estimator.estimate_batch(list(y_test))
        predictions_path = predictions_dir / "predictions.csv"
        with open(predictions_path, "w", encoding="utf-8") as f:
            f.write("true_class,predicted_class,true_calories_kcal_100g,predicted_calories_kcal_100g,correct\n")
            for t, p, tc, pc in zip(y_test, y_pred, true_calories, pred_calories):
                f.write(f"{t},{p},{tc},{pc},{int(t == p)}\n")

        self.logger.info("Artifacts saved: model, metrics.json, run_summary.json, predictions.csv")


def run_pipeline(cfg: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    try:
        pipeline = FoodCalorPipeline(cfg, logger)
        return pipeline.run()
    except FoodCalorieError:
        logger.exception("Pipeline failed with a known pipeline error")
        raise
    except Exception:
        logger.exception("Pipeline failed with an unexpected error")
        raise
