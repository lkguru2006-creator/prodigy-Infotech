# Customer Segmentation Pipeline

**Prodigy Infotech · Task-02**

Enterprise-grade K-means clustering pipeline that segments retail customers by purchase behaviour. Built to demonstrate end-to-end ML engineering best practices: centralised configuration, custom exception hierarchy, structured logging, fit/transform leakage prevention, and clean artifact management.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) add real Kaggle CSV at data/raw/Mall_Customers.csv
#    If absent, a realistic synthetic dataset is generated automatically.

# 3. Run the pipeline
python scripts/run_pipeline.py

# 4. Run tests
pytest tests/ -v                  # in any environment with pip
python run_tests_plain.py         # in network-restricted sandboxes

# 5. Explore the walkthrough notebook
jupyter notebook notebooks/customer_segmentation_walkthrough.ipynb
```

---

## Project Structure

```
customer-segmentation/
├── config/config.yaml                     # All tunable parameters
├── data/raw/Mall_Customers.csv            # Real CSV or synthetic placeholder
├── data/processed/customers_clustered.csv # Final labeled dataset
├── notebooks/customer_segmentation_walkthrough.ipynb
├── outputs/
│   ├── logs/pipeline.log
│   ├── metrics/  (metrics.json, cluster_profiles.json, run_summary.json)
│   ├── models/   (kmeans_model.joblib, scaler.joblib, feature_list.json)
│   └── plots/    (cluster_3d.png, cluster_pairplot.png, elbow_curve.png, ...)
├── scripts/run_pipeline.py                # Single entry point
├── src/
│   ├── data/         (loader.py, synthetic_data.py)
│   ├── features/     (feature_engineering.py)
│   ├── models/       (kmeans_model.py, evaluation.py)
│   ├── pipeline/     (orchestrator.py, persistence.py, visualization.py)
│   └── utils/        (config_loader.py, exceptions.py, logger.py)
├── tests/            (conftest.py + 6 test modules, pytest-structured)
├── run_tests_plain.py
├── pytest.ini
└── requirements.txt
```

---

## Pipeline Stages

| # | Stage | Module |
|---|-------|--------|
| 1 | Load & validate raw data | `src/data/loader.py` |
| 2 | Feature selection & StandardScaler | `src/features/feature_engineering.py` |
| 3 | K-means training (k=5, k-means++) | `src/models/kmeans_model.py` |
| 4 | Silhouette, Davies-Bouldin, elbow curve | `src/models/evaluation.py` |
| 5 | 2D/3D scatter, pairplot, bar chart, elbow plot | `src/pipeline/visualization.py` |
| 6 | Persist model, scaler, metrics, plots, labeled CSV | `src/pipeline/persistence.py` |

---

## Scoring New Customers

```python
import joblib, json, pandas as pd

model  = joblib.load("outputs/models/kmeans_model.joblib")
scaler = joblib.load("outputs/models/scaler.joblib")
with open("outputs/models/feature_list.json") as f:
    meta = json.load(f)

new_customers = pd.DataFrame({
    "Age": [28, 45],
    "Annual Income (k$)": [60, 90],
    "Spending Score (1-100)": [75, 20],
})
X_new = scaler.transform(new_customers[meta["feature_columns"]])
clusters = model.predict(X_new)
print("Assigned clusters:", clusters)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Fit/transform separation | Scaler reusable for new-customer scoring; prevents leakage |
| Custom exception hierarchy | Typed, catchable errors — no bare `Exception` |
| No `plt.show()` calls | Agg backend + save-only keeps headless/CI environments clean |
| Path resolution in config loader | Identical behaviour regardless of invocation directory |
| Synthetic data persisted to disk | Reproducibility; real CSV is a direct drop-in replacement |
