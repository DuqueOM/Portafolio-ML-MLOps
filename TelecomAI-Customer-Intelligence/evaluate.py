from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calcula métricas de clasificación binarias.

    Returns un diccionario con accuracy, precision, recall, f1, roc_auc (si hay proba).
    """
    results: Dict[str, float] = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            results["roc_auc"] = float(metrics.roc_auc_score(y_true, y_proba))
        except Exception as e:
            logger.warning("ROC AUC could not be computed: %s", e)
    return results


def save_metrics(metrics_dict: Dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info("Metrics saved to %s", path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path
) -> None:
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out_path)


def plot_roc_curve(
    y_true: np.ndarray, y_proba: Optional[np.ndarray], out_path: str | Path
) -> None:
    if y_proba is None:
        logger.warning("ROC curve cannot be plotted without probabilities.")
        return
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("ROC curve saved to %s", out_path)
