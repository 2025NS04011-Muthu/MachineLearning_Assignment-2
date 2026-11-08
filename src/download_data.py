"""Download the PROMISE NASA defect data sets and merge them into one CSV.

The Kaggle collection ``fracpete/promisedata`` requires authenticated access,
so this script fetches the very same PROMISE ARFF files (JM1, KC1, KC2, PC1)
from the public OpenML mirror.  All four share the 21 McCabe/Halstead static
code metrics, which makes them directly combinable.

Usage::

    python -m src.download_data
"""
from __future__ import annotations

import io
import logging
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.utils import (FEATURE_COLUMNS, RAW_DATA_PATH, SOURCE_COL, TARGET_COL,
                       ensure_dir, setup_logging)

LOGGER = logging.getLogger(__name__)

OPENML_URL = "https://www.openml.org/data/v1/download/{file_id}/{name}.arff"
PROMISE_FILES: Dict[str, int] = {"jm1": 53936, "kc1": 53950, "kc2": 53946, "pc1": 53951}

# Column names differ slightly between the ARFF files (case, spelling).
_ALIASES: Dict[str, str] = {c.lower(): c for c in FEATURE_COLUMNS}
_ALIASES["locodeandcomment"] = "locCodeAndComment"
_POSITIVE = {"true", "yes", "y", "1"}
_NEGATIVE = {"false", "no", "n", "0"}


def parse_arff(text: str) -> pd.DataFrame:
    """Parse a (simple, numeric) ARFF document into a DataFrame.

    Args:
        text: Raw ARFF file content.

    Returns:
        DataFrame whose ``?`` entries are converted to ``NaN``.
    """
    names: List[str] = []
    rows: List[str] = []
    in_data = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        lowered = stripped.lower()
        if lowered.startswith("@attribute"):
            match = re.match(r"@attribute\s+(\S+)", stripped, re.IGNORECASE)
            if match:
                names.append(match.group(1))
        elif lowered.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append(stripped)
    return pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=names, na_values="?")


def standardise_columns(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map project specific column names to the canonical feature names.

    Args:
        frame: Frame produced by :func:`parse_arff`.
        source: Name of the originating PROMISE project.

    Returns:
        Frame with canonical feature order, a textual ``defects`` target and a
        ``source_project`` column.
    """
    renamed = {col: _ALIASES.get(col.lower(), TARGET_COL) for col in frame.columns}
    frame = frame.rename(columns=renamed)
    target = frame[TARGET_COL].astype(str).str.strip().str.lower()
    unknown = set(target.unique()) - _POSITIVE - _NEGATIVE
    if unknown:
        raise ValueError(f"{source}: unexpected target values {sorted(unknown)}")
    frame[TARGET_COL] = target.isin(_POSITIVE).map({True: "true", False: "false"})
    frame[SOURCE_COL] = source.upper()
    return frame[FEATURE_COLUMNS + [TARGET_COL, SOURCE_COL]]


def fetch_project(name: str, file_id: int) -> Tuple[pd.DataFrame, int]:
    """Download one ARFF file and return the standardised frame and its row count."""
    url = OPENML_URL.format(file_id=file_id, name=name)
    LOGGER.info("Downloading %s from %s", name.upper(), url)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        text = response.read().decode("latin-1")
    frame = standardise_columns(parse_arff(text), name)
    return frame, len(frame)


def build_dataset(output: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Download every PROMISE project and persist the merged CSV.

    Args:
        output: Destination CSV path.

    Returns:
        The merged DataFrame.
    """
    frames = []
    for name, file_id in PROMISE_FILES.items():
        frame, count = fetch_project(name, file_id)
        LOGGER.info("%s: %d rows", name.upper(), count)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    ensure_dir(output.parent)
    merged.to_csv(output, index=False)
    LOGGER.info("Saved %d rows x %d columns to %s", *merged.shape, output)
    return merged


if __name__ == "__main__":
    setup_logging()
    try:
        build_dataset()
    except Exception:  # pragma: no cover - CLI guard
        LOGGER.exception("Could not build the PROMISE dataset")
        raise
