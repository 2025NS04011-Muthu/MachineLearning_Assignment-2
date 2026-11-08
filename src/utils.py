"""Shared constants and reusable helper functions for the defect-prediction project."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
MODEL_DIR: Path = ROOT_DIR / "model"
RAW_DATA_PATH: Path = DATA_DIR / "promise_defects.csv"
TEST_DATA_PATH: Path = ROOT_DIR / "test_data.csv"

SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"
IMPUTER_PATH: Path = MODEL_DIR / "imputer.pkl"
ENCODERS_PATH: Path = MODEL_DIR / "encoders.pkl"
METADATA_PATH: Path = MODEL_DIR / "metadata.json"
METRICS_PATH: Path = MODEL_DIR / "metrics.csv"

# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

TARGET_COL: str = "defects"
SOURCE_COL: str = "source_project"

FEATURE_COLUMNS: List[str] = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

CLASS_LABELS: Dict[int, str] = {0: "Non Defective", 1: "Defective"}

MODEL_FILES: Dict[str, str] = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}

METRIC_NAMES: List[str] = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "loc": "McCabe line count of code",
    "v(g)": "McCabe cyclomatic complexity",
    "ev(g)": "McCabe essential complexity",
    "iv(g)": "McCabe design complexity",
    "n": "Halstead total operators + operands",
    "v": "Halstead volume",
    "l": "Halstead program length",
    "d": "Halstead difficulty",
    "i": "Halstead intelligence",
    "e": "Halstead effort",
    "b": "Halstead error estimate",
    "t": "Halstead time estimator",
    "lOCode": "Halstead line count",
    "lOComment": "Halstead count of comment lines",
    "lOBlank": "Halstead count of blank lines",
    "locCodeAndComment": "Lines with both code and comment",
    "uniq_Op": "Unique operators",
    "uniq_Opnd": "Unique operands",
    "total_Op": "Total operators",
    "total_Opnd": "Total operands",
    "branchCount": "Branch count of the flow graph",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a consistent format.

    Args:
        level: Minimum severity that should be emitted.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Seed every random number generator used in the project.

    Args:
        seed: Seed value shared by Python, NumPy and hashing.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict[str, Any], path: Path) -> None:
    """Serialise ``payload`` to ``path`` as pretty-printed JSON."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its content as a dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays so they can be written as JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def model_path(model_name: str) -> Path:
    """Return the pickle path for a model display name.

    Raises:
        KeyError: If ``model_name`` is not one of the supported models.
    """
    if model_name not in MODEL_FILES:
        raise KeyError(f"Unknown model '{model_name}'. Choose from {list(MODEL_FILES)}")
    return MODEL_DIR / MODEL_FILES[model_name]
