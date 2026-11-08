"""Regenerate ``test_data.csv`` (the stratified hold-out split).

The split is deterministic (fixed seed), so running this script always
reproduces the exact rows used for the hold-out metrics reported in the README.

Usage::

    python -m src.generate_test_data
"""
from __future__ import annotations

import logging

from src.preprocessing import prepare_data
from src.train_models import write_test_data
from src.utils import RANDOM_STATE, set_global_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Create ``test_data.csv`` from the cleaned, stratified test partition."""
    setup_logging()
    set_global_seed(RANDOM_STATE)
    try:
        write_test_data(prepare_data())
    except Exception:
        LOGGER.exception("Could not generate test_data.csv")
        raise


if __name__ == "__main__":
    main()
