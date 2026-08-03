# Prodigy Infotech Internship Projects

This repository contains the projects completed during the Software Engineering & Machine Learning internship at **Prodigy Infotech**. Each project is fully self-contained, modular, and built using software engineering best practices such as centralized YAML configuration, structured logging, custom exception hierarchies, and fit/transform data leakage prevention.

## Repository Overview

The repository is structured as a collection of separate project folders, each addressing a specific task:

1. [**Task-02: Customer Segmentation Pipeline**](./customer-segmentation)
   * **Goal**: An enterprise-grade K-means clustering pipeline to segment retail customers based on purchasing behavior.
   * **Highlights**: Centrally configured YAML pipeline, silhouette and Davies-Bouldin evaluation, 2D/3D visualizations, and custom exception handling.
   * **Quickstart**: `python customer-segmentation/scripts/run_pipeline.py`

2. [**Task-03: SVM Cats vs Dogs Classifier**](./task03_svm_cats_dogs)
   * **Goal**: A binary image classifier to distinguish between cats and dogs using a Support Vector Machine (SVM) trained on HOG + Color Histogram features.
   * **Highlights**: Procedural synthetic generator fallback, Agg-backend visualization for confusion matrix, and complete unit/integration test coverage.
   * **Quickstart**: `python task03_svm_cats_dogs/main.py train`

3. [**Task-05: Food Recognition & Calorie Estimation**](./food_calorie_estimation)
   * **Goal**: A computer vision pipeline that recognizes food images and estimates their calorie content (kcal/100g) using scikit-learn classifiers (Random Forest / MLP) with HOG and color histogram features.
   * **Highlights**: Highly modular architecture with abstract base classes designed for seamless CNN pluggability.
   * **Quickstart**: `python food_calorie_estimation/main.py`

4. [**House Price Predictor**](./house-price-predictor)
   * **Goal**: A linear regression pipeline predicting house sale prices using features like square footage, bedrooms, and bathrooms.
   * **Highlights**: End-to-end regression validation, 5-fold cross-validation, imputation, and outlier handling.
   * **Quickstart**: `python house-price-predictor/scripts/run_pipeline.py`

5. [**Hand Gesture Recognition**](./hand_gesture_recognition)
   * **Goal**: A PyTorch CNN-based deep learning pipeline to recognize hand gestures from image frames.
   * **Highlights**: Early stopping, early model checkpointing, and self-contained predictor inference wrappers.
   * **Quickstart**: `python hand_gesture_recognition/main.py`

---

## Shared Engineering Guidelines

All projects in this repository adhere to the following design standards:
- **No Stray Prints**: All execution logs are channeled through a robust rotating file logger (`pipeline.log`) for production compatibility.
- **Config-Driven**: Magic numbers and hyperparameters are declared in local `config/config.yaml` files, never hardcoded.
- **No Data Leakage**: Data preprocessing scalers and feature engineers are fit exclusively on training data and reused for validation/test splits.
- **Offline / Sandbox Friendly**: Synthetic procedural generators are provided for all tasks, ensuring full execution and test passing out of the box even when raw data is not pre-downloaded.

For detailed run commands, dependencies, and testing strategies, please refer to the respective project's README file.
