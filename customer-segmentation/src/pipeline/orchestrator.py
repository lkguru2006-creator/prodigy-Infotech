"""
Pipeline orchestrator: wires together data loading, feature engineering,
model training, evaluation, visualization, and persistence into a single
cohesive, observable run. This is the only module that knows about the
full sequence of stages -- each stage module itself stays decoupled.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_raw_data
from src.features.feature_engineering import FeatureEngineer
from src.models.evaluation import (
    build_cluster_profiles,
    compute_davies_bouldin,
    compute_elbow_curve,
    compute_silhouette,
)
from src.models.kmeans_model import CustomerSegmentationModel
from src.pipeline import persistence, visualization
from src.utils.exceptions import CustomerSegmentationError


class SegmentationPipeline:
    """Orchestrates the full customer segmentation pipeline end to end."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.paths = config["paths"]

    def run(self) -> dict[str, Any]:
        """
        Execute the full pipeline. Returns the run summary dictionary that
        is also persisted to disk.
        """
        start_time = time.time()
        self.logger.info("=" * 70)
        self.logger.info(
            "Starting Customer Segmentation Pipeline (project: %s, v%s)",
            self.config["project"]["name"],
            self.config["project"]["version"],
        )
        self.logger.info("=" * 70)

        try:
            df = self._load_data()
            X_scaled, fe = self._engineer_features(df)
            model = self._train_model(X_scaled)
            labels = model.labels_
            metrics = self._evaluate(df, X_scaled, labels, model)
            self._visualize(df, labels, X_scaled)
            self._persist_artifacts(model, fe, metrics, df, labels)
            summary = self._build_summary(df, metrics, time.time() - start_time)
            persistence.save_json(
                summary,
                Path(self.paths["metrics_dir"]) / self.paths["run_summary_filename"],
                self.logger,
            )

            self.logger.info("=" * 70)
            self.logger.info(
                "Pipeline completed successfully in %.2f seconds.", time.time() - start_time
            )
            self.logger.info("=" * 70)
            return summary

        except CustomerSegmentationError as exc:
            self.logger.error("Pipeline failed with a known error: %s", exc)
            raise
        except Exception as exc:
            self.logger.exception("Pipeline failed with an unexpected error: %s", exc)
            raise

    # -- Stages ------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        self.logger.info("[1/6] Loading and validating raw data...")
        df = load_raw_data(self.config, self.logger)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> tuple[Any, FeatureEngineer]:
        self.logger.info("[2/6] Engineering features...")
        feat_cfg = self.config["features"]
        fe = FeatureEngineer(
            feature_columns=feat_cfg["clustering_features"],
            scale=feat_cfg["scale_features"],
        )
        X_scaled = fe.fit_transform(df, self.logger)
        return X_scaled, fe

    def _train_model(self, X_scaled: Any) -> CustomerSegmentationModel:
        self.logger.info("[3/6] Training K-means model...")
        model_cfg = self.config["model"]
        model = CustomerSegmentationModel(
            n_clusters=model_cfg["n_clusters"],
            init=model_cfg["init"],
            n_init=model_cfg["n_init"],
            max_iter=model_cfg["max_iter"],
            random_state=model_cfg["random_state"],
        )
        model.fit(X_scaled, self.logger)
        return model

    def _evaluate(
        self,
        df: pd.DataFrame,
        X_scaled: Any,
        labels: Any,
        model: CustomerSegmentationModel,
    ) -> dict[str, Any]:
        self.logger.info("[4/6] Evaluating clustering quality...")
        eval_cfg = self.config["evaluation"]
        metrics: dict[str, Any] = {
            "n_clusters": model.n_clusters,
            "inertia": round(model.inertia_, 4),
        }

        if eval_cfg["compute_silhouette"]:
            metrics["silhouette_score"] = round(compute_silhouette(X_scaled, labels), 4)
            metrics["davies_bouldin_score"] = round(compute_davies_bouldin(X_scaled, labels), 4)
            self.logger.info(
                "Silhouette score: %.4f | Davies-Bouldin index: %.4f",
                metrics["silhouette_score"],
                metrics["davies_bouldin_score"],
            )

        if eval_cfg["compute_elbow_curve"]:
            inertias = compute_elbow_curve(
                X_scaled,
                tuple(eval_cfg["elbow_k_range"]),
                self.config["model"]["random_state"],
                self.logger,
            )
            self._elbow_inertias = inertias  # stashed for visualization stage
        else:
            self._elbow_inertias = None

        profiles = build_cluster_profiles(
            df, labels, self.config["features"]["clustering_features"], self.logger
        )
        self._cluster_profiles = profiles

        return metrics

    def _visualize(self, df: pd.DataFrame, labels: Any, X_scaled: Any) -> None:
        self.logger.info("[5/6] Generating visualizations...")
        plots_dir = Path(self.paths["plots_dir"])
        feat_cols = self.config["features"]["clustering_features"]
        eval_cfg = self.config["evaluation"]

        visualization.plot_cluster_sizes(labels, plots_dir / "cluster_sizes.png")

        if len(feat_cols) >= 2:
            visualization.plot_cluster_scatter(
                df, labels, feat_cols[-2], feat_cols[-1], plots_dir / "cluster_scatter_2d.png"
            )

        if eval_cfg["generate_pairplot"]:
            visualization.plot_cluster_pairplot(df, labels, feat_cols, plots_dir / "cluster_pairplot.png")

        if eval_cfg["generate_3d_plot"]:
            visualization.plot_cluster_3d(df, labels, feat_cols, plots_dir / "cluster_3d.png")

        if getattr(self, "_elbow_inertias", None):
            visualization.plot_elbow_curve(self._elbow_inertias, plots_dir / "elbow_curve.png")

        self.logger.info("All plots saved to '%s'.", plots_dir)

    def _persist_artifacts(
        self,
        model: CustomerSegmentationModel,
        fe: FeatureEngineer,
        metrics: dict[str, Any],
        df: pd.DataFrame,
        labels: Any,
    ) -> None:
        self.logger.info("[6/6] Persisting artifacts...")
        model_dir = Path(self.paths["model_dir"])
        metrics_dir = Path(self.paths["metrics_dir"])

        persistence.save_artifact(model.model, model_dir / self.paths["model_filename"], self.logger)
        if fe.scaler is not None:
            persistence.save_artifact(fe.scaler, model_dir / self.paths["scaler_filename"], self.logger)
        fe.save_feature_list(model_dir / self.paths["feature_list_filename"])

        persistence.save_json(metrics, metrics_dir / self.paths["metrics_filename"], self.logger)
        persistence.save_json(
            self._cluster_profiles, metrics_dir / self.paths["cluster_profile_filename"], self.logger
        )

        # Final labeled dataset (the clustering equivalent of a submission file)
        output_df = df.copy()
        output_df["Cluster"] = labels
        processed_path = Path(self.config["data"]["processed_path"])
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(processed_path, index=False)
        self.logger.info("Labeled dataset written to '%s'.", processed_path)

    def _build_summary(
        self, df: pd.DataFrame, metrics: dict[str, Any], elapsed_seconds: float
    ) -> dict[str, Any]:
        return {
            "project": self.config["project"]["name"],
            "version": self.config["project"]["version"],
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "n_customers": int(len(df)),
            "clustering_features": self.config["features"]["clustering_features"],
            "metrics": metrics,
            "cluster_profiles": self._cluster_profiles,
            "random_seed": self.config["project"]["random_seed"],
        }
