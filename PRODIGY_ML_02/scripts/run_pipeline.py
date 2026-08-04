#!/usr/bin/env python3
"""
Single entry point for the Customer Segmentation Pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config/config.yaml

Run from the project root. This script wires together configuration
loading, logging setup, and pipeline execution, and prints a clean
human-readable summary on success -- no stack traces, no warnings
leaking to stdout on the happy path.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Ensure the project root is on sys.path so `src` imports resolve when this
# script is invoked directly (python scripts/run_pipeline.py) rather than
# as a module.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy third-party FutureWarning/UserWarning chatter (e.g. from
# sklearn KMeans n_init defaults) so console output stays clean. Pipeline
# errors are never suppressed -- only non-actionable library warnings.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.pipeline.orchestrator import SegmentationPipeline  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.exceptions import CustomerSegmentationError  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the customer segmentation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def print_summary(summary: dict) -> None:
    """Print a clean, human-readable run summary to stdout."""
    metrics = summary["metrics"]
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE - RUN SUMMARY")
    print("=" * 60)
    print(f"  Project          : {summary['project']} (v{summary['version']})")
    print(f"  Customers        : {summary['n_customers']}")
    print(f"  Clusters (k)     : {metrics['n_clusters']}")
    print(f"  Features used    : {', '.join(summary['clustering_features'])}")
    print(f"  Silhouette score : {metrics.get('silhouette_score', 'N/A')}")
    print(f"  Davies-Bouldin   : {metrics.get('davies_bouldin_score', 'N/A')}")
    print(f"  Inertia          : {metrics['inertia']}")
    print(f"  Elapsed time     : {summary['elapsed_seconds']}s")
    print("-" * 60)
    print("  Cluster sizes:")
    for name, profile in summary["cluster_profiles"].items():
        cid = name.replace("cluster_", "")
        print(f"    Cluster {cid}: {profile['size']} customers ({profile['pct_of_total']}%)")
    print("=" * 60)
    print("Artifacts written to: outputs/models, outputs/metrics, outputs/plots")
    print("Labeled dataset:      data/processed/customers_clustered.csv")
    print("=" * 60 + "\n")


def main() -> int:
    args = parse_args()

    try:
        config = load_config(args.config)
    except CustomerSegmentationError as exc:
        print(f"\n[CONFIGURATION ERROR] {exc}\n", file=sys.stderr)
        return 1

    logging_cfg = config["logging"]
    logger = get_logger(
        name="customer_segmentation",
        log_dir=config["paths"]["log_dir"],
        log_filename=config["paths"]["log_filename"],
        level=logging_cfg["level"],
        max_bytes=logging_cfg["max_bytes"],
        backup_count=logging_cfg["backup_count"],
        fmt=logging_cfg["format"],
    )

    try:
        pipeline = SegmentationPipeline(config, logger)
        summary = pipeline.run()
        print_summary(summary)
        return 0
    except CustomerSegmentationError as exc:
        print(f"\n[PIPELINE ERROR] {exc}\n", file=sys.stderr)
        print("See outputs/logs/pipeline.log for full details.\n", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n[UNEXPECTED ERROR] {exc}\n", file=sys.stderr)
        print("See outputs/logs/pipeline.log for full details.\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
