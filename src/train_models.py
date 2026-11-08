"""Train, evaluate and persist the five classifiers.

Usage::

    python -m src.train_models           # train with the stored best hyper-parameters
    python -m src.train_models --tune    # re-run the cross-validated search first
"""
from __future__ import annotations

import argparse
import logging
import platform
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import compare_models, positive_scores
from src.preprocessing import PreparedData, prepare_data, random_oversample
from src.utils import (CLASS_LABELS, ENCODERS_PATH, FEATURE_COLUMNS, IMPUTER_PATH, METADATA_PATH,
                       METRICS_PATH, MODEL_DIR, MODEL_FILES, RANDOM_STATE, RAW_DATA_PATH,
                       SCALER_PATH, TARGET_COL, TEST_DATA_PATH, TEST_SIZE, ensure_dir,
                       model_path, save_json, set_global_seed, setup_logging)

LOGGER = logging.getLogger(__name__)

# Best values found by ``tune_hyperparameters`` (3-fold CV, ROC-AUC, training data only).
BEST_PARAMS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {"C": 10},
    "Decision Tree": {"max_depth": 4, "min_samples_leaf": 20},
    "KNN": {"n_neighbors": 101, "weights": "uniform"},
    "Naive Bayes": {"var_smoothing": 0.1},
    "SVM": {"C": 1, "gamma": 0.01},
}

PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    # depth starts at 4 so the tree uses enough features for a meaningful importance chart
    "Decision Tree": {"max_depth": [4, 5, 6, 8], "min_samples_leaf": [5, 20, 50, 100]},
    "KNN": {"n_neighbors": [15, 31, 51, 101], "weights": ["uniform", "distance"]},
    "Naive Bayes": {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
    "SVM": {"C": [0.1, 1], "gamma": ["scale", 0.01]},
}


def build_models(
    random_state: int = RANDOM_STATE, params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Create the five (untrained) classifiers.

    Args:
        random_state: Seed for every stochastic estimator.
        params: Hyper-parameters per model name (defaults to :data:`BEST_PARAMS`).
    """
    p = params or BEST_PARAMS
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, random_state=random_state, **p["Logistic Regression"]),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, **p["Decision Tree"]),
        "KNN": KNeighborsClassifier(**p["KNN"]),
        "Naive Bayes": GaussianNB(**p["Naive Bayes"]),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state, **p["SVM"]),
    }


def tune_hyperparameters(
    data: PreparedData, folds: int = 3, random_state: int = RANDOM_STATE
) -> Dict[str, Dict[str, Any]]:
    """Cross-validated grid search (ROC-AUC) on the training partition only.

    Minority over-sampling is applied inside every fold so validation folds never
    contain duplicated training rows; the hold-out test set is never touched.

    Returns:
        Best parameters per model name.
    """
    skf = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(data.X_train_unbalanced, data.y_train_unbalanced))
    best: Dict[str, Dict[str, Any]] = {}
    for name, grid in PARAM_GRIDS.items():
        results = []
        for candidate in ParameterGrid(grid):
            candidate_params = {**BEST_PARAMS, name: candidate}
            aucs = []
            for fit_idx, val_idx in splits:
                X_fit, y_fit = random_oversample(
                    data.X_train_unbalanced[fit_idx], data.y_train_unbalanced[fit_idx], random_state)
                model = build_models(random_state, candidate_params)[name].set_params(
                    **({"probability": False} if name == "SVM" else {})).fit(X_fit, y_fit)
                aucs.append(roc_auc_score(data.y_train_unbalanced[val_idx],
                                          positive_scores(model, data.X_train_unbalanced[val_idx])))
            results.append((float(np.mean(aucs)), candidate))
        score, best[name] = max(results, key=lambda item: item[0])
        LOGGER.info("Tuned %-20s CV AUC %.4f %s", name, score, best[name])
    return best


def train_models(
    data: PreparedData,
    random_state: int = RANDOM_STATE,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fit every classifier on the same (scaled, balanced) training data.

    Args:
        data: Output of :func:`src.preprocessing.prepare_data`.
        random_state: Seed for the estimators.
        params: Optional hyper-parameters (defaults to :data:`BEST_PARAMS`).

    Returns:
        Mapping ``display name -> fitted model``.
    """
    fitted: Dict[str, Any] = {}
    for name, model in build_models(random_state, params).items():
        LOGGER.info("Training %s on %d rows", name, len(data.y_train))
        model.fit(data.X_train, data.y_train)
        fitted[name] = model
    return fitted


def save_artifacts(models: Dict[str, Any], data: PreparedData, metrics: pd.DataFrame) -> None:
    """Persist models, preprocessing objects, metrics and metadata.

    Args:
        models: Fitted classifiers.
        data: Prepared data (provides scaler, imputer, encoders and report).
        metrics: Hold-out comparison table.
    """
    ensure_dir(MODEL_DIR)
    for name, model in models.items():
        joblib.dump(model, model_path(name), compress=3)
    joblib.dump(data.scaler, SCALER_PATH)
    joblib.dump(data.imputer, IMPUTER_PATH)
    joblib.dump(data.encoders, ENCODERS_PATH)
    metrics.to_csv(METRICS_PATH, index=False)

    raw = pd.read_csv(RAW_DATA_PATH)
    save_json(
        {
            "source": "PROMISE repository (NASA MDP: JM1, KC1, KC2, PC1)",
            "raw_rows": int(len(raw)),
            "projects": raw["source_project"].value_counts().to_dict() if "source_project" in raw else {},
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COL,
            "class_labels": CLASS_LABELS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "preprocessing": data.report,
            "hyperparameters": {n: m.get_params() for n, m in models.items()},
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        },
        METADATA_PATH,
    )
    LOGGER.info("Artifacts written to %s", MODEL_DIR)


def write_test_data(data: PreparedData, path=TEST_DATA_PATH) -> None:
    """Write the hold-out split (raw, unscaled metrics + target) as ``test_data.csv``."""
    data.test_frame.to_csv(path, index=False)
    LOGGER.info("Wrote %d test rows to %s", len(data.test_frame), path)


def run_training(tune: bool = False) -> pd.DataFrame:
    """Execute the whole pipeline and return the hold-out comparison table.

    Args:
        tune: Re-run the cross-validated hyper-parameter search (slow, ~2 min)
            instead of using :data:`BEST_PARAMS`.

    Raises:
        FileNotFoundError: If the PROMISE CSV has not been downloaded.
    """
    set_global_seed(RANDOM_STATE)
    data = prepare_data()
    params = tune_hyperparameters(data) if tune else None
    models = train_models(data, params=params)
    metrics = compare_models(models, data.X_test, data.y_test)
    save_artifacts(models, data, metrics)
    write_test_data(data)
    return metrics


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Train the five defect-prediction models.")
    parser.add_argument("--tune", action="store_true", help="re-run the cross-validated search")
    args = parser.parse_args()
    setup_logging()
    try:
        metrics = run_training(tune=args.tune)
    except Exception:
        LOGGER.exception("Training failed")
        raise
    print("\n" + metrics.round(4).to_string(index=False))
    missing = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"Model files missing after training: {missing}")


if __name__ == "__main__":
    main()
