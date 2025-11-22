#!/usr/bin/env python3
"""
Drift Detection Script using Evidently
Detects data drift and model performance degradation
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(reference_path: str, current_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference and current datasets."""
    logger.info(f"Loading reference data from: {reference_path}")
    reference_data = pd.read_csv(reference_path)

    logger.info(f"Loading current data from: {current_path}")
    current_data = pd.read_csv(current_path)

    return reference_data, current_data


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str = "Exited",
    numerical_features: list = None,
    categorical_features: list = None,
) -> Report:
    """
    Detect data drift using Evidently.

    Args:
        reference_data: Historical/training data
        current_data: New/production data
        target_column: Target variable name
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names

    Returns:
        Evidently Report object
    """
    logger.info("Detecting data drift...")

    # Create drift report using Evidently preset API
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    # Let Evidently infer column roles automatically based on the data
    report.run(reference_data=reference_data, current_data=current_data)

    logger.info("Drift detection completed")
    return report


def extract_drift_metrics(report: Report) -> Dict:
    """Extract key drift metrics from Evidently report."""
    result = report.as_dict()

    drift_metrics = {
        "timestamp": datetime.now().isoformat(),
        "dataset_drift": result["metrics"][0]["result"]["dataset_drift"],
        "number_of_drifted_columns": result["metrics"][0]["result"]["number_of_drifted_columns"],
        "share_of_drifted_columns": result["metrics"][0]["result"]["share_of_drifted_columns"],
        "drift_by_columns": {},
    }

    # Extract per-column drift
    drift_by_columns = result["metrics"][0]["result"].get("drift_by_columns", {})
    for column, drift_info in drift_by_columns.items():
        drift_metrics["drift_by_columns"][column] = {
            "drift_detected": drift_info.get("drift_detected", False),
            "drift_score": drift_info.get("drift_score", None),
        }

    return drift_metrics


def check_drift_threshold(drift_metrics: Dict, threshold: float = 0.5) -> bool:
    """
    Check if drift exceeds threshold.

    Args:
        drift_metrics: Dictionary with drift metrics
        threshold: Maximum acceptable share of drifted columns

    Returns:
        True if drift exceeds threshold, False otherwise
    """
    share_drifted = drift_metrics.get("share_of_drifted_columns", 0)
    logger.info(f"Share of drifted columns: {share_drifted:.2%}")

    if share_drifted > threshold:
        logger.warning(f"⚠️  DRIFT ALERT: {share_drifted:.2%} of columns drifted (threshold: {threshold:.2%})")
        return True
    else:
        logger.info(f"✓ Drift within acceptable limits ({share_drifted:.2%} < {threshold:.2%})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Detect data drift using Evidently")
    parser.add_argument("--reference", required=True, help="Path to reference dataset (CSV)")
    parser.add_argument("--current", required=True, help="Path to current dataset (CSV)")
    parser.add_argument("--output-html", default="reports/drift_report.html", help="Output HTML report path")
    parser.add_argument("--output-json", default="reports/drift_metrics.json", help="Output JSON metrics path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Drift threshold (0-1)")
    parser.add_argument("--target", default="Exited", help="Target column name")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_html).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    reference_data, current_data = load_data(args.reference, args.current)

    logger.info(f"Reference data shape: {reference_data.shape}")
    logger.info(f"Current data shape: {current_data.shape}")

    # Define features (automatically detect from columns)
    numerical_features = reference_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = reference_data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove target from features
    if args.target in numerical_features:
        numerical_features.remove(args.target)
    if args.target in categorical_features:
        categorical_features.remove(args.target)

    logger.info(f"Numerical features: {len(numerical_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")

    # Detect drift
    report = detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        target_column=args.target,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # Save HTML report
    logger.info(f"Saving HTML report to: {args.output_html}")
    report.save_html(args.output_html)

    # Extract and save metrics
    drift_metrics = extract_drift_metrics(report)
    logger.info(f"Saving metrics to: {args.output_json}")
    with open(args.output_json, "w") as f:
        json.dump(drift_metrics, f, indent=2)

    # Check threshold
    drift_exceeded = check_drift_threshold(drift_metrics, threshold=args.threshold)

    # Print summary
    print("\n" + "=" * 60)
    print("DRIFT DETECTION SUMMARY")
    print("=" * 60)
    print(f"Dataset Drift Detected: {drift_metrics['dataset_drift']}")
    print(f"Drifted Columns: {drift_metrics['number_of_drifted_columns']}")
    print(f"Share of Drifted Columns: {drift_metrics['share_of_drifted_columns']:.2%}")
    print(f"Threshold Exceeded: {drift_exceeded}")
    print(f"HTML Report: {args.output_html}")
    print(f"JSON Metrics: {args.output_json}")
    print("=" * 60 + "\n")

    # Exit with error code if drift exceeds threshold
    if drift_exceeded:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
