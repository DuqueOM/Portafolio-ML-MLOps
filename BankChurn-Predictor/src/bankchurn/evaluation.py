"""Model evaluation and metrics for churn prediction.

This module provides comprehensive evaluation functionality including:
- Standard classification metrics
- ROC curves and calibration plots
- Confusion matrices
- Fairness metrics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for churn prediction.

    Parameters
    ----------
    model : object
        Trained model with predict and predict_proba methods.
    preprocessor : object
        Fitted preprocessor for feature transformation.

    Attributes
    ----------
    metrics_ : dict
        Computed evaluation metrics.
    """

    def __init__(self, model: Any, preprocessor: Any) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.metrics_: dict[str, float] = {}

    @classmethod
    def from_files(cls, model_path: str | Path, preprocessor_path: str | Path) -> ModelEvaluator:
        """Load model and preprocessor from disk.

        Parameters
        ----------
        model_path : str or Path
            Path to saved model.
        preprocessor_path : str or Path
            Path to saved preprocessor.

        Returns
        -------
        evaluator : ModelEvaluator
            Initialized evaluator with loaded artifacts.
        """
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
        return cls(model, preprocessor)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate model performance on test data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : Series
            True labels.
        output_path : str or Path, optional
            Path to save evaluation results.

        Returns
        -------
        metrics : dict
            Evaluation metrics including:
            - accuracy, precision, recall, f1
            - roc_auc
            - confusion_matrix
            - classification_report
        """
        # Transform features
        X_transformed = self.preprocessor.transform(X)

        # Predictions
        y_pred = self.model.predict(X_transformed)

        # Probabilities (if available)
        try:
            y_proba = self.model.predict_proba(X_transformed)
            has_proba = True
        except AttributeError:
            y_proba = None
            has_proba = False

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        }

        # AUC if binary and probabilities available
        if has_proba and len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, y_proba[:, 1])

            # ROC curve
            fpr, tpr, thresholds = roc_curve(y, y_proba[:, 1])
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }

        self.metrics_ = metrics

        # Log summary
        logger.info("Evaluation Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        if "roc_auc" in metrics:
            logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        # Save if requested
        if output_path:
            self._save_results(output_path)

        return metrics

    def _save_results(self, output_path: str | Path) -> None:
        """Save evaluation results to JSON.

        Parameters
        ----------
        output_path : str or Path
            Path to save results.
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.metrics_, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")

    def compute_fairness_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: list[str],
    ) -> dict[str, Any]:
        """Compute fairness metrics across sensitive features.

        Parameters
        ----------
        X : DataFrame
            Feature matrix including sensitive features.
        y : Series
            True labels.
        sensitive_features : list of str
            Column names of sensitive features (e.g., ["gender", "age_group"]).

        Returns
        -------
        fairness_metrics : dict
            Fairness metrics per sensitive feature including:
            - performance by group
            - disparate impact ratios
        """
        # Transform for prediction
        X_transformed = self.preprocessor.transform(X)
        y_pred = self.model.predict(X_transformed)

        fairness_metrics = {}

        for feature in sensitive_features:
            if feature not in X.columns:
                logger.warning(f"Sensitive feature '{feature}' not found in data")
                continue

            groups = X[feature].unique()
            group_metrics = {}

            for group in groups:
                mask = X[feature] == group
                if mask.sum() == 0:
                    continue

                y_true_group = y[mask]
                y_pred_group = y_pred[mask]

                group_metrics[str(group)] = {
                    "count": int(mask.sum()),
                    "accuracy": float(accuracy_score(y_true_group, y_pred_group)),
                    "precision": float(
                        precision_score(y_true_group, y_pred_group, average="weighted", zero_division=0)
                    ),
                    "recall": float(recall_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                    "f1": float(f1_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                }

            # Compute disparate impact (ratio of positive rates)
            positive_rates = {}
            for group in groups:
                mask = X[feature] == group
                if mask.sum() > 0:
                    positive_rate = (y_pred[mask] == 1).mean()
                    positive_rates[str(group)] = float(positive_rate)

            if len(positive_rates) >= 2:
                rates = list(positive_rates.values())
                disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 0.0
            else:
                disparate_impact = 1.0

            fairness_metrics[feature] = {
                "groups": group_metrics,
                "disparate_impact": float(disparate_impact),
            }

            logger.info(f"Fairness for '{feature}': Disparate Impact = {disparate_impact:.4f}")

        return fairness_metrics
