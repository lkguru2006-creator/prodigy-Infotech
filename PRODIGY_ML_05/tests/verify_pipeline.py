"""Plain-Python verification script (no pytest dependency) for sandboxed/offline
environments. Equivalent checks to test_pipeline.py, run with:

    python tests/verify_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.config_loader import load_config
from src.features.feature_extractor import FoodFeatureExtractor
from src.models.calorie_estimator import CalorieEstimator
from src.models.sklearn_classifier import SklearnFoodClassifier
from src.utils.logger import get_logger


def run_checks() -> int:
    failures = 0
    cfg = load_config()
    logger = get_logger("verify_logger", cfg)

    def check(name: str, condition: bool) -> None:
        nonlocal failures
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")
        if not condition:
            failures += 1

    check("config paths are absolute", cfg["data"]["raw_dir"].startswith("/"))

    estimator = CalorieEstimator(cfg, logger)
    check("calorie lookup: known class", estimator.estimate("pizza") == 266)
    check(
        "calorie lookup: unknown class falls back to default",
        estimator.estimate("not_a_real_class") == cfg["calorie_lookup"]["default"],
    )

    fe = FoodFeatureExtractor(cfg, logger)
    fake_images = np.random.randint(0, 255, size=(5, 64, 64, 3), dtype=np.uint8)
    features = fe.fit_transform(fake_images)
    check("feature extractor fit_transform shape", features.shape[0] == 5)

    fe2 = FoodFeatureExtractor(cfg, logger)
    raised = False
    try:
        fe2.transform(fake_images)
    except Exception:
        raised = True
    check("feature extractor raises before fit()", raised)

    clf = SklearnFoodClassifier(cfg, logger)
    raised = False
    try:
        clf.predict(np.zeros((1, 10)))
    except Exception:
        raised = True
    check("classifier raises before fit()", raised)

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_checks())
