"""Read-only loaders for the project's generated output files.

Every loader:
  * resolves paths relative to the project root (never absolute hard-codes),
  * returns an empty dict/DataFrame (not raising) when a file is missing or
    malformed so the UI can show a graceful "unavailable" message, and
  * never loads the raw Spark performance file into pandas -- only the small
    aggregated outputs and the test-prediction CSV parts are read.

Caching of loaded data is handled by ``dashboard.app`` via
``st.cache_data``; this module keeps the loaders pure and easy to unit-test.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger("dashboard.data_loader")

# Project root = parent of the dashboard package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUTS: Path = PROJECT_ROOT / "outputs"
METRICS_DIR: Path = OUTPUTS / "metrics"
TABLES_DIR: Path = OUTPUTS / "tables"
PREDICTIONS_DIR: Path = OUTPUTS / "predictions" / "test_predictions_csv"


def project_root() -> Path:
    """Absolute path to the project root (for display in the UI)."""
    return PROJECT_ROOT


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame, returning empty on any error."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        log.warning("Empty CSV: %s", path)
        return pd.DataFrame()
    except Exception as exc:  # malformed / unreadable
        log.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def load_model_metrics(path: Optional[Path] = None) -> dict:
    """``outputs/metrics/model_metrics.json`` as a dict (empty if missing)."""
    path = Path(path) if path is not None else METRICS_DIR / "model_metrics.json"
    if not path.exists():
        log.warning("Missing model_metrics.json at %s", path)
        return {}
    try:
        return _read_json(path)
    except Exception as exc:
        log.warning("Could not parse %s: %s", path, exc)
        return {}


def load_model_metrics_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """``outputs/metrics/model_metrics.csv`` (single-row summary)."""
    path = Path(path) if path is not None else METRICS_DIR / "model_metrics.csv"
    if not path.exists():
        log.warning("Missing model_metrics.csv at %s", path)
        return pd.DataFrame()
    return _load_csv(path)


def load_validation_threshold_metrics(path: Optional[Path] = None) -> pd.DataFrame:
    """Both candidate models x thresholds 0.05-0.50 on the validation set."""
    path = Path(path) if path is not None else METRICS_DIR / "validation_threshold_metrics.csv"
    if not path.exists():
        log.warning("Missing validation_threshold_metrics.csv at %s", path)
        return pd.DataFrame()
    return _load_csv(path)


def load_data_quality(path: Optional[Path] = None) -> dict:
    """``outputs/tables/data_quality.json`` (row count, duplicates, missingness)."""
    path = Path(path) if path is not None else TABLES_DIR / "data_quality.json"
    if not path.exists():
        log.warning("Missing data_quality.json at %s", path)
        return {}
    try:
        return _read_json(path)
    except Exception as exc:
        log.warning("Could not parse %s: %s", path, exc)
        return {}


def load_target_distribution(path: Optional[Path] = None) -> pd.DataFrame:
    """``outputs/tables/target_distribution.csv`` (default 0/1 -> count)."""
    path = Path(path) if path is not None else TABLES_DIR / "target_distribution.csv"
    if not path.exists():
        log.warning("Missing target_distribution.csv at %s", path)
        return pd.DataFrame()
    df = _load_csv(path)
    if not df.empty and {"default", "count"}.issubset(df.columns):
        df["default"] = pd.to_numeric(df["default"], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
    return df


def load_descriptive_statistics(path: Optional[Path] = None) -> pd.DataFrame:
    """Spark ``describe()`` output (summary rows x feature columns)."""
    path = Path(path) if path is not None else TABLES_DIR / "descriptive_statistics.csv"
    if not path.exists():
        log.warning("Missing descriptive_statistics.csv at %s", path)
        return pd.DataFrame()
    return _load_csv(path)


def load_balance_proxy(path: Optional[Path] = None) -> pd.DataFrame:
    """``income_risk_summary.csv``.

    IMPORTANT: the Freddie Mac public sample has NO borrower income field.
    This file groups loans by original unpaid principal balance (UPB). It is a
    *loan-balance* proxy, not an income analysis.
    """
    path = Path(path) if path is not None else TABLES_DIR / "income_risk_summary.csv"
    if not path.exists():
        log.warning("Missing income_risk_summary.csv at %s", path)
        return pd.DataFrame()
    df = _load_csv(path)
    if not df.empty:
        for col in ("borrower_count", "average_predicted_default_probability",
                    "observed_default_rate", "average_credit_score"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_test_predictions(directory: Optional[Path] = None) -> pd.DataFrame:
    """Concatenate every ``part-*.csv`` in the Spark prediction output directory.

    Spark writes one header per part file, so each file is read independently and
    concatenated. CRC / ``_SUCCESS`` markers are skipped by the ``part-*.csv``
    glob.
    """
    directory = Path(directory) if directory is not None else PREDICTIONS_DIR
    if not directory.exists():
        log.warning("Missing predictions directory %s", directory)
        return pd.DataFrame()
    parts = sorted(directory.glob("part-*.csv"))
    if not parts:
        log.warning("No part-*.csv files in %s", directory)
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for part in parts:
        frame = _load_csv(part)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Coerce the columns the dashboard relies on; leave unknowns untouched.
    for col in ("default", "prediction"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("predicted_default_probability", "credit_score", "dti",
                "ltv", "cltv", "upb"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def balance_bands_from_predictions(predictions: pd.DataFrame,
                                    upb_col: str = "upb") -> pd.DataFrame:
    """Build loan-balance (UPB) bands directly from the test predictions.

    Returns one row per band with borrower_count, average predicted default
    probability (historical-period model estimate) and observed default rate
    (realised outcome). Explicitly labelled as loan-balance, never income.
    """
    if predictions.empty or upb_col not in predictions.columns:
        return pd.DataFrame()
    df = predictions.copy()
    df["balance_band"] = pd.cut(
        df[upb_col],
        bins=[-float("inf"), 100000, 250000, 500000, float("inf")],
        labels=["<$100k", "$100k-$250k", "$250k-$500k", "$500k+"]).astype("object")
    grouped = df.groupby("balance_band", observed=False, dropna=False).agg(
        borrower_count=("loan_sequence_number", "count") if "loan_sequence_number" in df.columns else (upb_col, "count"),
        average_predicted_default_probability=("predicted_default_probability", "mean"),
        observed_default_rate=("default", "mean"),
        average_credit_score=("credit_score", "mean"),
    ).reset_index()
    return grouped
