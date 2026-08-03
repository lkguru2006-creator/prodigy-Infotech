# House Price Predictor

Enterprise-grade linear regression pipeline that predicts house sale prices
from square footage, bedroom count, and bathroom count (plus supporting
structural features: overall quality, year built, garage capacity).

Built on the schema of the Kaggle **House Prices - Advanced Regression
Techniques** competition.

## Architecture

```
house-price-predictor/
├── config/
│   └── config.yaml              # All tunable parameters — no hardcoded values in code
├── data/
│   ├── raw/                     # train.csv / test.csv (real Kaggle files or synthetic fallback)
│   └── processed/                # Validated/snapshotted data used for training
├── src/house_price_predictor/
│   ├── data/
│   │   ├── ingestion.py          # Loads real CSVs if present, else generates synthetic data
│   │   ├── validation.py         # Schema + sanity checks (fail fast on bad data)
│   │   └── synthetic_generator.py# Drop-in stand-in dataset, same schema as real Kaggle data
│   ├── features/
│   │   └── engineering.py        # Imputation, TotalSF/TotalBath/HouseAge, outlier clipping
│   ├── models/
│   │   ├── trainer.py            # Scaled LinearRegression + k-fold cross-validation
│   │   ├── evaluator.py          # RMSE / MAE / R2 / MAPE
│   │   ├── persistence.py        # Save/load model, scaler, feature list, metrics
│   │   └── inference.py          # Predict on new raw data using saved artifacts
│   ├── pipeline/
│   │   ├── training_pipeline.py  # Orchestrates ingestion → validation → features → train → eval → save
│   │   └── prediction_pipeline.py# Orchestrates load model → predict test set → write submission.csv
│   └── utils/
│       ├── config.py             # Typed YAML config loader
│       ├── logger.py             # Structured, rotating-file + console logging (no print statements)
│       └── exceptions.py         # Project-specific exception hierarchy
├── scripts/
│   └── run_pipeline.py           # Single entry point: training + prediction
├── tests/                        # Unit + integration tests (pytest)
├── artifacts/
│   ├── models/                   # Persisted model, scaler, feature list
│   └── reports/                  # metrics.json, run_summary.json, submission.csv
└── logs/                         # Rotating pipeline.log
```

## Design principles

- **Single source of truth for config.** Every threshold, filename, and
  hyperparameter lives in `config/config.yaml`. Nothing is hardcoded in
  source files.
- **No leakage.** Imputation medians and outlier bounds are learned only on
  the training split; the same fitted `FeatureEngineer` instance is reused
  for the holdout set and the Kaggle test set.
- **No raw `print()` calls.** All progress and diagnostics go through a
  structured logger (console + rotating file at `logs/pipeline.log`).
  Final numeric results are written as JSON/CSV artifacts, not printed.
- **Fail fast, fail clearly.** A custom exception hierarchy
  (`DataValidationError`, `ModelTrainingError`, etc.) ensures errors are
  caught at the layer they originate in and surfaced with context, not
  silently swallowed.
- **Real-data ready.** The synthetic generator exists purely so the
  pipeline is runnable without the actual Kaggle files. Drop the real
  `train.csv` / `test.csv` into `data/raw/` and the pipeline automatically
  prefers them — zero code changes required.

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Add the real Kaggle dataset

Download `train.csv` and `test.csv` from the
[Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place them in `data/raw/`. If absent, the pipeline auto-generates a
schema-compatible synthetic dataset on first run.

### 3. Run the full pipeline

```bash
python scripts/run_pipeline.py
```

This will:
1. Ingest data (real or synthetic)
2. Validate schema and sanity checks
3. Engineer features (`TotalSF`, `TotalBath`, `HouseAge`, imputation, outlier handling)
4. Train a scaled linear regression model with 5-fold cross-validation
5. Evaluate on a held-out split (RMSE, MAE, R², MAPE)
6. Persist the model, scaler, and feature list to `artifacts/models/`
7. Generate `artifacts/reports/submission.csv` for the test set

Use `--skip-prediction` to only train/evaluate without generating predictions.

### 4. Run tests

```bash
pip install pytest pytest-cov
pytest
```

## Output artifacts

| File | Description |
|---|---|
| `artifacts/models/linear_regression_model.joblib` | Trained estimator |
| `artifacts/models/feature_scaler.joblib` | Fitted `StandardScaler` |
| `artifacts/models/feature_list.json` | Ordered feature names used at training time |
| `artifacts/reports/metrics.json` | Train/holdout RMSE, MAE, R², MAPE + CV stats |
| `artifacts/reports/run_summary.json` | Full run summary (metrics + feature list) |
| `artifacts/reports/submission.csv` | Kaggle-format predictions (`Id`, `SalePrice`) |
| `logs/pipeline.log` | Rotating structured log of every pipeline run |
