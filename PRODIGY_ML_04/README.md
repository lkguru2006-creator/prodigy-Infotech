# Hand Gesture Recognition Pipeline

Enterprise-style, modular pipeline for classifying hand gestures from image
data, built against the structure of the
[LeapGestRecog dataset](https://www.kaggle.com/gti-upm/leapgestrecog).

## Why a synthetic generator?

The raw Kaggle CSV/image archive wasn't provided at build time. Rather than
block on that, `src/data/synthetic_generator.py` procedurally generates a
dataset with the **exact same directory contract** as the real one:

```
data/raw/leapGestRecog/<subject 00-09>/<gesture class folder>/<frame>.png
```

Because every downstream component (`dataset.py`, training, evaluation)
only depends on that directory contract — not on how the pixels were
produced — swapping in real data requires **no code changes**:

1. Download & extract the Kaggle archive so it lands at `data/raw/leapGestRecog/...`
2. Set `synthetic.enabled: false` in `config/config.yaml`
3. Re-run the pipeline

## Project layout

```
config/config.yaml         Single source of truth for all parameters
src/
  config_loader.py         YAML -> typed Config object
  data/
    synthetic_generator.py Procedural dataset generator (drop-in replaceable)
    dataset.py              Indexing, stratified split, PyTorch Dataset, transforms
  models/cnn_model.py        CNN architecture
  training/trainer.py        Training loop, early stopping, checkpointing
  evaluation/evaluator.py    Metrics + JSON report generation
  inference/predictor.py     Self-contained inference wrapper
  utils/                     Logging (no print statements anywhere) & seeding
scripts/
  generate_data.py           CLI: generate/refresh synthetic data
  train.py                   CLI: train end-to-end
  evaluate.py                CLI: evaluate best checkpoint on test split
  predict.py                 CLI: predict a single image
main.py                      Orchestrates generate -> train -> evaluate
artifacts/
  models/                    Saved checkpoints (best model only)
  logs/                      pipeline.log (structured, timestamped)
  reports/                   evaluation_report.json
```

## Usage

```bash
pip install -r requirements.txt

# Full pipeline (generate synthetic data -> train -> evaluate)
python main.py

# Or run stages individually
python scripts/generate_data.py
python scripts/train.py
python scripts/evaluate.py
python scripts/predict.py --image path/to/some_gesture.png
```

## Design notes

- **No stray `print()` calls.** All runtime output goes through the
  centralized logger (`src/utils/logger.py`), which writes structured,
  leveled, timestamped lines to both stdout and `artifacts/logs/pipeline.log`.
  The only exception is `scripts/predict.py`, which intentionally prints a
  final JSON payload as its CLI output contract.
- **Config-driven.** No magic numbers scattered through the codebase —
  everything (image size, split ratios, hyperparameters, paths) lives in
  `config/config.yaml`.
- **Self-contained checkpoints.** The saved `.pt` file embeds class names,
  image size, and architecture params, so `predictor.py` can reconstruct
  the model and preprocessing without needing the training config at
  inference time.
- **Stratified splitting** ensures every class is proportionally
  represented in train/val/test.
- **Idempotent data generation** — re-running `generate_data.py` won't
  duplicate data unless `--force` is passed.
