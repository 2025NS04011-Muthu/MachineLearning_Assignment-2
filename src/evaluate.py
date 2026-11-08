"""Metric computation and model comparison helpers."""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src.utils import CLASS_LABELS, METRIC_NAMES

LOGGER = logging.getLogger(__name__)


def positive_scores(model: Any, features: np.ndarray) -> np.ndarray:
    """Return the probability (or decision score) of the defective class.

    Args:
        model: Fitted scikit-learn classifier.
        features: Scaled feature matrix.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute the six assignment metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_score: Scores for the positive class (used for AUC).

    Returns:
        Mapping with Accuracy, AUC, Precision, Recall, F1 and MCC.
    """
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        LOGGER.warning("AUC undefined because only one class is present")
        auc = float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate_model(model: Any, features: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """Score one model and return metrics plus raw predictions.

    Returns:
        Dict with ``metrics``, ``y_pred`` and ``y_score``.
    """
    y_pred = model.predict(features)
    y_score = positive_scores(model, features)
    return {
        "metrics": compute_metrics(y_true, y_pred, y_score),
        "y_pred": y_pred,
        "y_score": y_score,
    }


def compare_models(models: Mapping[str, Any], features: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Evaluate every model and build the comparison table.

    Args:
        models: Mapping ``display name -> fitted model``.
        features: Scaled evaluation features.
        y_true: Ground-truth labels.

    Returns:
        DataFrame with columns ``ML Model, Accuracy, AUC, Precision, Recall, F1, MCC``.
    """
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, features, y_true)["metrics"]
        rows.append({"ML Model": name, **metrics})
        LOGGER.info("%-20s %s", name, {k: round(v, 4) for k, v in metrics.items()})
    return pd.DataFrame(rows, columns=["ML Model", *METRIC_NAMES])


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the 2x2 confusion matrix (rows = actual, columns = predicted)."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return precision / recall / F1 / support per class as a DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[CLASS_LABELS[0], CLASS_LABELS[1]],
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).T
    frame["support"] = frame["support"].astype(int)
    return frame


def get_roc_data(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(fpr, tpr, auc)`` for a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(roc_auc_score(y_true, y_score))


def rank_models(table: pd.DataFrame) -> pd.DataFrame:
    """Rank models by MCC, then F1, then AUC (all descending).

    Args:
        table: Output of :func:`compare_models`.

    Returns:
        Sorted copy with a 1-based ``Rank`` column in front.
    """
    ranked = table.sort_values(["MCC", "F1", "AUC"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    return ranked
