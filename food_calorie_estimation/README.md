# Task-05: Food Recognition & Calorie Estimation

Enterprise-grade pipeline that classifies food images and maps predictions to
estimated calorie content (kcal/100g), for the Prodigy Infotech internship.

**Dataset schema:** [Kaggle Food-101](https://www.kaggle.com/dansbecker/food-101)
(folder-per-class images). Real data is drop-in replaceable.

## Why sklearn, not a CNN framework
This environment has no network access and no TensorFlow/PyTorch installed. The
model layer is built behind an abstract interface (`src/models/base_model.py`),
with a scikit-learn backend (HOG + color-histogram features → RandomForest/MLP)
that runs fully offline today. To use a real CNN in a GPU-enabled deployment,
implement `BaseFoodClassifier` in a new `src/models/keras_cnn_model.py` and set
`model.backend: "keras_cnn"` in config — no pipeline code changes required.

## Architecture
```
data layer      -> src/data/            (synthetic generator + loader/splitter)
features layer  -> src/features/        (HOG + color hist, fit/transform separated)
models layer    -> src/models/          (abstract interface, sklearn backend, calorie lookup)
pipeline layer  -> src/pipeline/        (orchestration)
utils           -> src/utils/           (config loader, rotating logger, exceptions)
entry point     -> main.py
```

## Using real Food-101 data
1. Download from Kaggle and arrange as `data/raw/<class_name>/*.jpg`
2. Set `synthetic.enabled: false` in `config/config.yaml`
3. Update `synthetic.class_names` / `calorie_lookup` to match the classes you use
4. Run `python main.py`

## Run
```bash
pip install -r requirements.txt
python main.py                          # uses config/config.yaml by default
python main.py --config path/to/cfg.yaml
```

## Test
```bash
pytest tests/                           # networked environments
python tests/verify_pipeline.py         # offline/sandboxed environments
```

## Outputs (deterministic paths, absolute-resolved from project root)
- `outputs/models/food_classifier.joblib`
- `outputs/metrics/metrics.json`, `outputs/metrics/run_summary.json`
- `outputs/predictions/predictions.csv` (true/predicted class + calorie estimates)
- `outputs/logs/pipeline.log` (rotating file handler, no print statements)

## Key design notes
- All paths resolved to absolute, project-root-anchored paths in `config_loader.py`
  to avoid artifacts scattering when invoked from outside the project directory.
- Feature scaler fit only on training data; test data only ever transformed.
- Calorie estimation uses a class→calorie lookup table (auditable, deterministic)
  rather than a regression head, since the classifier output is a discrete class.
