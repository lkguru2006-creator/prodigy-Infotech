# Task-03: SVM Cats vs Dogs Classifier

Enterprise-style ML pipeline for the Kaggle
[Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data) binary image classification
task, built around a support-vector machine trained on HOG + color-histogram features.

## Using real Kaggle data
Download `train.zip` / `test1.zip` from Kaggle, unzip, and drop the files in as-is:
- `data/raw/train/cat.N.jpg`, `data/raw/train/dog.N.jpg`
- `data/raw/test1/<id>.jpg`

No code changes required — the synthetic generator only runs when `train/` is empty.

## Quickstart
```bash
pip install -r requirements.txt

python main.py train                                       # full training run
python main.py predict --input data/raw/test1 --output submission.csv
python main.py generate-data                                # force-regenerate synthetic data
```

## Architecture
```
config/config.yaml          Central YAML config (paths, hyperparams, feature/split settings)
src/utils/                  config_loader (root-anchored absolute paths), logger, exceptions
src/data/                   synthetic_generator.py, data_loader.py (discovery + splits)
src/features/               feature_extractor.py — HOG + color histogram, fit/transform separated
src/models/                 svm_model.py — SVC wrapper (train/evaluate/persist/reload)
src/pipeline/                training_pipeline.py, prediction_pipeline.py — orchestration
main.py                     CLI: train / predict / generate-data
tests/                      pytest suite + plain-Python fallback (tests/run_all_checks.py)
notebooks/walkthrough.ipynb 21-cell notebook walkthrough of the same src/ package
```

## Design principles
- **No raw `print()`** — structured rotating loggers only (`artifacts/logs/`).
- **No `plt.show()`** — Matplotlib Agg backend; figures saved to `artifacts/figures/`.
- **Absolute, root-anchored paths** — resolved once in `config_loader.py`, avoiding CWD-dependent bugs.
- **Fit/transform separation** — the feature scaler is fit only on the training split.
- **Custom exception hierarchy** — `DataError`, `FeatureExtractionError`, `ModelError`, `PredictionError`.
- **Deterministic artifacts** — model, scaler, metrics JSON, run summary, confusion-matrix figure,
  logs, and submission CSV all land at fixed paths under `artifacts/` and `outputs/`.

## Verifying
```bash
python -m pytest tests/ -v            # networked environments
python tests/run_all_checks.py        # sandboxed / offline fallback (no pytest install needed)
```

## Output artifacts (after `python main.py train`)
- `artifacts/models/svm_classifier.joblib`, `feature_pipeline.joblib`
- `artifacts/metrics/run_summary.json` (train/val/test sizes, metrics, config snapshot)
- `artifacts/figures/confusion_matrix.png`
- `artifacts/logs/*.log`
- `outputs/submission.csv` (after `python main.py predict`), format: `id,label` (0=cat, 1=dog)
