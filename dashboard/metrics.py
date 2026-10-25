"""Metric helpers recomputed from the saved test predictions.

These functions are pure (take a DataFrame or arrays, return values) so they
are trivially unit-testable. They reproduce curves/confusion matrices from
existing prediction rows -- they never retrain and never invent data.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# np.trapz was removed in NumPy 2.0 (renamed to np.trapezoid). Support both.
try:  # NumPy >= 2.0
    from numpy import trapezoid as _trapezoid
except ImportError:  # NumPy < 2.0
    from numpy import trapz as _trapezoid


def confusion_matrix(df: pd.DataFrame, pred_col: str = "prediction",
                     label_col: str = "default") -> dict:
    """TP / TN / FP / FN from a binary prediction column vs the actual label."""
    if df is None or df.empty:
        return {}
    if pred_col not in df.columns or label_col not in df.columns:
        return {}
    work = df[[pred_col, label_col]].dropna()
    if work.empty:
        return {}
    pred = work[pred_col].astype(int)
    label = work[label_col].astype(int)
    tp = int(((label == 1) & (pred == 1)).sum())
    tn = int(((label == 0) & (pred == 0)).sum())
    fp = int(((label == 0) & (pred == 1)).sum())
    fn = int(((label == 1) & (pred == 0)).sum())
    return {"true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn}


def rates(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Accuracy / precision / recall / F1 from a confusion matrix."""
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / total if total else 0.0
    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


def roc_curve(df: pd.DataFrame, prob_col: str = "predicted_default_probability",
              label_col: str = "default"
              ) -> Tuple[np.ndarray, np.ndarray, float]:
    """ROC curve (fpr, tpr) and area under it via the trapezoidal rule.

    Returns (fpr_array, tpr_array, auc). Empty inputs -> ([], [], 0.0).
    """
    if df is None or df.empty or prob_col not in df.columns or label_col not in df.columns:
        return np.array([]), np.array([]), 0.0
    work = df[[prob_col, label_col]].dropna().sort_values(prob_col, ascending=False)
    if work.empty:
        return np.array([]), np.array([]), 0.0
    y = work[label_col].to_numpy().astype(int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return np.array([]), np.array([]), 0.0
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tp / positives])
    fpr = np.concatenate([[0.0], fp / negatives])
    auc = float(_trapezoid(tpr, fpr))
    return fpr, tpr, auc


def pr_curve(df: pd.DataFrame, prob_col: str = "predicted_default_probability",
             label_col: str = "default"
             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Precision-recall curve (recall, precision) and trapezoidal area.

    Returns (recall_array, precision_array, auc). Empty -> ([], [], 0.0).
    """
    if df is None or df.empty or prob_col not in df.columns or label_col not in df.columns:
        return np.array([]), np.array([]), 0.0
    work = df[[prob_col, label_col]].dropna().sort_values(prob_col, ascending=False)
    if work.empty:
        return np.array([]), np.array([]), 0.0
    y = work[label_col].to_numpy().astype(int)
    positives = int(y.sum())
    if positives == 0:
        return np.array([]), np.array([]), 0.0
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    denom = np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = tp / denom
    # Prepend a (recall=0, precision=1) point for a cleaner curve.
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    auc = float(_trapezoid(precision, recall))
    return recall, precision, auc


def default_rate_by_group(df: pd.DataFrame, group_col: str,
                          label_col: str = "default") -> pd.DataFrame:
    """Observed historical default rate and count per categorical group."""
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()
    work = df[[group_col, label_col]].dropna()
    if work.empty:
        return pd.DataFrame()
    grouped = (work.groupby(group_col, dropna=False)
               .agg(loan_count=(label_col, "count"),
                    observed_defaults=(label_col, "sum"),
                    observed_default_rate=(label_col, "mean"))
               .reset_index())
    return grouped


def numeric_band_default_rate(df: pd.DataFrame, num_col: str,
                              bins, labels,
                              label_col: str = "default") -> pd.DataFrame:
    """Bin a numeric feature and compute the historical default rate per band."""
    if df is None or df.empty or num_col not in df.columns:
        return pd.DataFrame()
    work = df[[num_col, label_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame()
    work["band"] = pd.cut(work[num_col], bins=bins, labels=labels)
    grouped = (work.groupby("band", observed=False, dropna=False)
               .agg(loan_count=(label_col, "count"),
                    observed_defaults=(label_col, "sum"),
                    observed_default_rate=(label_col, "mean"),
                    average_value=(num_col, "mean"))
               .reset_index())
    return grouped
