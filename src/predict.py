"""Load persisted artifacts and generate predictions."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from src.preprocessing import extract_target, transform_features
from src.utils import (CLASS_LABELS, ENCODERS_PATH, IMPUTER_PATH, METADATA_PATH, METRICS_PATH,
                       MODEL_FILES, SCALER_PATH, TARGET_COL, load_json, model_path)

LOGGER = logging.getLogger(__name__)


@dataclass
class Artifacts:
    """Everything the application needs at inference time."""

    models: Dict[str, Any]
    scaler: Any
    imputer: Any
    encoders: Dict[str, Any]
    metadata: Dict[str, Any]
    training_metrics: pd.DataFrame


def load_artifacts() -> Artifacts:
    """Load models, preprocessing objects and metadata from ``model/``.

    Raises:
        FileNotFoundError: If any artifact is missing.
        InconsistentVersionWarning: Escalated to an error so callers can retrain
            when the installed scikit-learn differs from the training one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        models = {name: joblib.load(model_path(name)) for name in MODEL_FILES}
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
    metadata = load_json(METADATA_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    LOGGER.info("Loaded %d models and preprocessing artifacts", len(models))
    return Artifacts(models, scaler, imputer, encoders, metadata, metrics)


def ensure_artifacts() -> Artifacts:
    """Load artifacts, retraining from the bundled dataset if they are unusable.

    This keeps the app working on any host (for example Streamlit Community
    Cloud) even when pickles were produced with a different scikit-learn.
    """
    try:
        return load_artifacts()
    except Exception as exc:  # noqa: BLE001 - any failure triggers a rebuild
        LOGGER.warning("Artifacts unusable (%s). Retraining models...", exc)
        from src.train_models import run_training

        run_training()
        return load_artifacts()


def prepare_inputs(df: pd.DataFrame, artifacts: Artifacts) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Impute and scale an uploaded frame and extract its target if present.

    Args:
        df: Frame that already passed :func:`src.preprocessing.validate_schema`.
        artifacts: Loaded artifacts.

    Returns:
        ``(scaled_features, y_true_or_None)``.
    """
    features = transform_features(df, artifacts.imputer, artifacts.scaler)
    return features, extract_target(df)


def predict_dataframe(
    df: pd.DataFrame, model_name: str, artifacts: Artifacts
) -> pd.DataFrame:
    """Predict defect status for every row of ``df``.

    Args:
        df: Uploaded frame with the 21 metric columns.
        model_name: One of :data:`src.utils.MODEL_FILES`.
        artifacts: Loaded artifacts.

    Returns:
        Copy of ``df`` extended with ``Predicted``, ``Predicted_Label`` and
        ``Defect_Probability`` columns.
    """
    model = artifacts.models[model_name]
    features, _ = prepare_inputs(df, artifacts)
    predicted = model.predict(features)
    result = df.copy().reset_index(drop=True)
    result["Predicted"] = predicted.astype(int)
    result["Predicted_Label"] = result["Predicted"].map(CLASS_LABELS)
    if hasattr(model, "predict_proba"):
        result["Defect_Probability"] = model.predict_proba(features)[:, 1].round(4)
    if TARGET_COL in result.columns:
        result = result.rename(columns={TARGET_COL: "Actual"})
    leading = [c for c in ("Predicted_Label", "Defect_Probability", "Actual", "Predicted")
               if c in result.columns]
    return result[leading + [c for c in result.columns if c not in leading]]
