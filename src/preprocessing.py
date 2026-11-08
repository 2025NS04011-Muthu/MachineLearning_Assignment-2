"""Data loading, cleaning and feature preparation for defect prediction.

The pipeline is deliberately leak-free: duplicates and invalid rows are removed
before the split, while imputation, scaling and class balancing are fitted on
the training partition only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import (CLASS_LABELS, FEATURE_COLUMNS, RANDOM_STATE, RAW_DATA_PATH,
                       SOURCE_COL, TARGET_COL, TEST_SIZE)

LOGGER = logging.getLogger(__name__)

_POSITIVE = {"true", "yes", "y", "1", "1.0", "defective"}
_NEGATIVE = {"false", "no", "n", "0", "0.0", "non defective", "non-defective"}


# --------------------------------------------------------------------------- #
# Loading and target handling
# --------------------------------------------------------------------------- #
def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the merged PROMISE CSV.

    Args:
        path: Location of the CSV file.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the file is missing (run ``python -m src.download_data``).
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m src.download_data` first."
        )
    frame = pd.read_csv(path)
    LOGGER.info("Loaded %d rows x %d columns from %s", *frame.shape, path)
    return frame


def normalize_target(series: pd.Series) -> pd.Series:
    """Convert a target column to binary integers (0 = non defective, 1 = defective).

    Accepts booleans, 0/1 numbers and strings such as ``true/false``, ``yes/no``.

    Raises:
        ValueError: If a value cannot be interpreted.
    """
    if series.dtype == bool:
        return series.astype(int)
    text = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(text.unique()) - _POSITIVE - _NEGATIVE)
    if unknown:
        raise ValueError(f"Unsupported target values: {unknown[:5]}")
    return text.isin(_POSITIVE).astype(int)


def coerce_numeric(frame: pd.DataFrame, columns: List[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Return a copy where ``columns`` are numeric (invalid text becomes ``NaN``)."""
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _any_feature_negative(df: pd.DataFrame) -> pd.Series:
    return (df[FEATURE_COLUMNS] < 0).any(axis=1)


INVALID_RULES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "negative metric value": _any_feature_negative,
    "module without code (loc < 1)": lambda d: d["loc"] < 1,
    "complexity below 1": lambda d: (d["v(g)"] < 1) | (d["ev(g)"] < 1)
    | (d["iv(g)"] < 1) | (d["branchCount"] < 1),
    "ev(g) or iv(g) greater than v(g)": lambda d: (d["ev(g)"] > d["v(g)"])
    | (d["iv(g)"] > d["v(g)"]),
    "v(g) greater than loc": lambda d: d["v(g)"] > d["loc"],
    "lOCode greater than loc": lambda d: d["lOCode"] > d["loc"],
    "unique tokens exceed total tokens": lambda d: (d["uniq_Op"] > d["total_Op"])
    | (d["uniq_Opnd"] > d["total_Opnd"]),
    "Halstead length != operators + operands": lambda d: (
        d["n"] - (d["total_Op"] + d["total_Opnd"])
    ).abs() > 1,
    "Halstead level l above 1": lambda d: d["l"] > 1,
}


def find_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every data-quality rule.

    Args:
        df: Numeric feature frame.

    Returns:
        Boolean DataFrame (one column per rule); ``NaN`` comparisons count as valid.
    """
    return pd.DataFrame({name: rule(df).fillna(False) for name, rule in INVALID_RULES.items()})


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw frame: types, target, duplicates and invalid records.

    Missing *feature* values are kept here and imputed after the split so that
    no information from the test set leaks into the imputer.

    Args:
        df: Raw merged frame.

    Returns:
        Tuple ``(clean_frame, report)`` where ``report`` documents every step.
    """
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {missing_cols}")

    report: Dict[str, Any] = {"rows_loaded": int(len(df))}
    frame = coerce_numeric(df)
    report["missing_cells"] = int(frame[FEATURE_COLUMNS].isna().sum().sum())

    frame = frame[frame[TARGET_COL].notna()].copy()
    frame[TARGET_COL] = normalize_target(frame[TARGET_COL])

    subset = FEATURE_COLUMNS + [TARGET_COL]
    before = len(frame)
    frame = frame.drop_duplicates(subset=subset).reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(frame))

    flags = find_invalid_records(frame)
    report["invalid_by_rule"] = {k: int(v) for k, v in flags.sum().items() if v}
    invalid_mask = flags.any(axis=1)
    report["invalid_removed"] = int(invalid_mask.sum())
    frame = frame.loc[~invalid_mask].reset_index(drop=True)

    report["rows_final"] = int(len(frame))
    report["missing_cells_final"] = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    LOGGER.info(
        "Cleaning: %d loaded -> %d duplicates, %d invalid removed -> %d rows",
        report["rows_loaded"], report["duplicates_removed"],
        report["invalid_removed"], report["rows_final"],
    )
    return frame, report


# --------------------------------------------------------------------------- #
# Splitting, imputation, scaling, balancing
# --------------------------------------------------------------------------- #
def split_data(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split of the cleaned frame.

    Returns:
        ``X_train, X_test, y_train, y_test`` (original indices preserved).
    """
    return train_test_split(
        frame[FEATURE_COLUMNS + ([SOURCE_COL] if SOURCE_COL in frame else [])],
        frame[TARGET_COL],
        test_size=test_size,
        stratify=frame[TARGET_COL],
        random_state=random_state,
    )


def random_oversample(
    features: np.ndarray, target: np.ndarray, random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance classes by resampling the minority class with replacement.

    Args:
        features: 2-D training matrix.
        target: Binary labels.
        random_state: Seed for reproducibility.

    Returns:
        Shuffled, balanced ``(features, target)``.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(target, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return features, target
    majority = counts.max()
    index_parts = []
    for cls, count in zip(classes, counts):
        idx = np.flatnonzero(target == cls)
        extra = rng.choice(idx, size=majority - count, replace=True) if count < majority else []
        index_parts.append(np.concatenate([idx, extra]).astype(int))
    order = rng.permutation(np.concatenate(index_parts))
    return features[order], target[order]


@dataclass
class PreparedData:
    """Container for everything produced by :func:`prepare_data`."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_unbalanced: np.ndarray
    y_train_unbalanced: np.ndarray
    test_frame: pd.DataFrame
    imputer: SimpleImputer
    scaler: StandardScaler
    encoders: Dict[str, Any]
    report: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


def build_encoders() -> Dict[str, Any]:
    """Return the target encoder used across training and inference."""
    return {
        "target_column": TARGET_COL,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_to_int": {"false": 0, "true": 1},
        "int_to_label": dict(CLASS_LABELS),
    }


def log_transform(values: np.ndarray) -> np.ndarray:
    """Apply ``log1p`` to tame the heavy right skew of the code metrics.

    Negative inputs (possible in user uploads) are clipped to zero first.
    """
    return np.log1p(np.clip(values, 0, None))


def prepare_data(
    path: Path = RAW_DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    balance: bool = True,
) -> PreparedData:
    """Run the full preprocessing pipeline.

    Steps: load, clean, stratified split, median imputation, log1p transform,
    standard scaling
    (both fitted on train only) and random over-sampling of the training set.

    Args:
        path: Source CSV.
        test_size: Fraction reserved for testing.
        random_state: Seed shared by every random step.
        balance: Whether to over-sample the minority class in the training set.

    Returns:
        A populated :class:`PreparedData`.
    """
    frame, report = clean_data(load_dataset(path))
    X_train, X_test, y_train, y_test = split_data(frame, test_size, random_state)

    imputer = SimpleImputer(strategy="median").fit(X_train[FEATURE_COLUMNS])
    train_imputed = pd.DataFrame(imputer.transform(X_train[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS)
    test_imputed = pd.DataFrame(imputer.transform(X_test[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS)

    scaler = StandardScaler().fit(log_transform(train_imputed.to_numpy()))
    train_scaled = scaler.transform(log_transform(train_imputed.to_numpy()))
    test_scaled = scaler.transform(log_transform(test_imputed.to_numpy()))

    train_target = y_train.to_numpy()
    if balance:
        balanced_X, balanced_y = random_oversample(train_scaled, train_target, random_state)
    else:
        balanced_X, balanced_y = train_scaled, train_target

    test_frame = X_test[FEATURE_COLUMNS].copy()
    test_frame[TARGET_COL] = y_test.to_numpy()

    report.update(
        {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_rows_after_balancing": int(len(balanced_y)),
            "class_distribution_clean": {
                CLASS_LABELS[int(k)]: int(v) for k, v in frame[TARGET_COL].value_counts().sort_index().items()
            },
            "class_distribution_train": {
                CLASS_LABELS[int(k)]: int(v) for k, v in y_train.value_counts().sort_index().items()
            },
            "class_distribution_test": {
                CLASS_LABELS[int(k)]: int(v) for k, v in y_test.value_counts().sort_index().items()
            },
            "balancing": "random over-sampling of the minority class (train only)" if balance else "none",
            "transform": "log1p on all metrics, then StandardScaler",
        }
    )
    return PreparedData(
        X_train=balanced_X,
        y_train=balanced_y,
        X_test=test_scaled,
        y_test=y_test.to_numpy(),
        X_train_unbalanced=train_scaled,
        y_train_unbalanced=train_target,
        test_frame=test_frame.reset_index(drop=True),
        imputer=imputer,
        scaler=scaler,
        encoders=build_encoders(),
        report=report,
    )


# --------------------------------------------------------------------------- #
# Inference-time helpers
# --------------------------------------------------------------------------- #
@dataclass
class SchemaReport:
    """Outcome of validating an uploaded CSV."""

    valid: bool
    missing_columns: List[str]
    extra_columns: List[str]
    has_target: bool
    non_numeric_cells: int
    missing_cells: int
    messages: List[str]


def _abbreviate(names: List[str], limit: int = 6) -> str:
    """Join column names, truncating long lists as ``a, b, ... (+N more)``."""
    if len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])} ... (+{len(names) - limit} more)"


def validate_schema(df: pd.DataFrame) -> SchemaReport:
    """Check that an uploaded frame matches the training schema.

    The 21 metric columns are mandatory; the ``defects`` target is optional
    (without it only predictions, not metrics, can be produced).

    Args:
        df: Uploaded frame.

    Returns:
        A :class:`SchemaReport` describing problems and warnings.
    """
    messages: List[str] = []
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    allowed = set(FEATURE_COLUMNS) | {TARGET_COL, SOURCE_COL}
    extra = [c for c in df.columns if c not in allowed]
    has_target = TARGET_COL in df.columns

    non_numeric = 0
    missing_cells = 0
    present = [c for c in FEATURE_COLUMNS if c in df.columns]
    if present:
        original_na = df[present].isna().sum().sum()
        converted = df[present].apply(pd.to_numeric, errors="coerce")
        non_numeric = int(converted.isna().sum().sum() - original_na)
        missing_cells = int(original_na)

    valid = not missing and len(df) > 0
    if df.empty:
        messages.append("The file contains no rows.")
    if missing:
        if len(missing) == len(FEATURE_COLUMNS):
            messages.append(
                "This file does not look like PROMISE software-metrics data - none of the "
                f"{len(FEATURE_COLUMNS)} required metric columns were found. Please upload the project's "
                "test_data.csv (or use the bundled sample)."
            )
        else:
            messages.append(f"Missing required columns: {_abbreviate(missing)}")
    if extra and not missing:
        messages.append(f"Ignoring unexpected columns: {_abbreviate(extra)}")
    if not has_target and valid:
        messages.append(f"Column '{TARGET_COL}' not found - metrics will be unavailable.")
    if non_numeric:
        messages.append(f"{non_numeric} non-numeric cells will be treated as missing.")
    if missing_cells:
        messages.append(f"{missing_cells} missing cells will be imputed with training medians.")

    if has_target and valid:
        try:
            normalize_target(df[TARGET_COL].dropna())
        except ValueError as exc:
            valid = False
            messages.append(str(exc))

    return SchemaReport(valid, missing, extra, has_target, non_numeric, missing_cells, messages)


def transform_features(
    df: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler
) -> np.ndarray:
    """Impute, log-transform and scale raw metric columns exactly as in training.

    Args:
        df: Frame containing all ``FEATURE_COLUMNS``.
        imputer: Fitted imputer.
        scaler: Fitted scaler.

    Returns:
        Scaled feature matrix ready for any of the classifiers.
    """
    numeric = coerce_numeric(df[FEATURE_COLUMNS])
    imputed = pd.DataFrame(imputer.transform(numeric), columns=FEATURE_COLUMNS)
    return scaler.transform(log_transform(imputed.to_numpy()))


def extract_target(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return the binary target of ``df`` or ``None`` if absent / incomplete."""
    if TARGET_COL not in df.columns or df[TARGET_COL].isna().any():
        return None
    return normalize_target(df[TARGET_COL]).to_numpy()
