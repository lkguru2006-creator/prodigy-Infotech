"""Plain-Python verification script (no pytest dependency required).

Sandboxed / network-restricted environments cannot always `pip install
pytest`. This script re-implements the same assertions as the pytest
suite in tests/test_*.py using only the standard library + already
vendored project dependencies, so the pipeline can still be verified
locally. Run with: python tests/run_all_checks.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.data_loader import discover_train_records, train_val_test_split
from src.features.feature_extractor import ImageFeaturePipeline
from src.models.svm_model import SVMClassifier
from src.utils.exceptions import DataError, FeatureExtractionError, ModelError

PASSED = 0
FAILED = 0


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"[PASS] {name}")
    except Exception as exc:  # noqa: BLE001
        FAILED += 1
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()


def test_feature_pipeline_shapes():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(6, 64, 64, 3), dtype=np.uint8)
    pipeline = ImageFeaturePipeline()
    features = pipeline.fit_transform(images)
    assert features.shape[0] == 6


def test_feature_pipeline_transform_before_fit():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(2, 64, 64, 3), dtype=np.uint8)
    pipeline = ImageFeaturePipeline()
    try:
        pipeline.transform(images)
        raise AssertionError("Expected FeatureExtractionError")
    except FeatureExtractionError:
        pass


def test_svm_fit_predict():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (20, 5)), rng.normal(5, 1, (20, 5))])
    y = np.array([0] * 20 + [1] * 20)
    clf = SVMClassifier()
    clf.fit(x, y)
    preds = clf.predict(x)
    assert preds.shape == y.shape


def test_svm_predict_before_fit():
    clf = SVMClassifier()
    try:
        clf.predict(np.zeros((2, 5)))
        raise AssertionError("Expected ModelError")
    except ModelError:
        pass


def test_discover_train_records():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        (tmp_dir / "cat.0.jpg").write_bytes(b"fake")
        (tmp_dir / "dog.0.jpg").write_bytes(b"fake")
        records = discover_train_records(tmp_dir)
        assert sorted(r.label for r in records) == [0, 1]
    finally:
        shutil.rmtree(tmp_dir)


def test_split_sizes():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(100, 8, 8, 3), dtype=np.uint8)
    labels = np.array([0, 1] * 50)
    splits = train_val_test_split(images, labels, test_size=0.2, val_size=0.1,
                                    stratify=True, random_seed=42)
    total = len(splits["x_train"]) + len(splits["x_val"]) + len(splits["x_test"])
    assert total == 100


if __name__ == "__main__":
    check("feature_pipeline_shapes", test_feature_pipeline_shapes)
    check("feature_pipeline_transform_before_fit", test_feature_pipeline_transform_before_fit)
    check("svm_fit_predict", test_svm_fit_predict)
    check("svm_predict_before_fit", test_svm_predict_before_fit)
    check("discover_train_records", test_discover_train_records)
    check("split_sizes", test_split_sizes)

    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(0 if FAILED == 0 else 1)
