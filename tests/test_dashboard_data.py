"""Tests for dashboard data loading and metric calculations.

These use synthetic fixtures (not the pipeline's real outputs) so they pass
whether or not the Spark pipeline has been run. They also verify graceful
handling of missing files and multi-part Spark CSV directories.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import data_loader, metrics


# --------------------------------------------------------------------------- #
# Metric calculations (pure functions)
# --------------------------------------------------------------------------- #
def test_confusion_matrix_counts():
    df = pd.DataFrame({"default": [1, 1, 0, 0, 1, 0],
                       "prediction": [1, 0, 1, 0, 1, 0]})
    cm = metrics.confusion_matrix(df)
    # Positives (default=1) at rows 0,1,4 with preds 1,0,1 -> TP=2, FN=1.
    # Negatives (default=0) at rows 2,3,5 with preds 1,0,0 -> FP=1, TN=2.
    assert cm == {"true_positive": 2, "true_negative": 2,
                  "false_positive": 1, "false_negative": 1}


def test_confusion_matrix_empty_dataframe():
    assert metrics.confusion_matrix(pd.DataFrame()) == {}
    assert metrics.confusion_matrix(None) == {}


def test_rates_calculation():
    r = metrics.rates(tp=3, tn=90, fp=7, fn=10)
    assert r["accuracy"] == pytest.approx(93 / 110)
    assert r["precision"] == pytest.approx(3 / 10)
    assert r["recall"] == pytest.approx(3 / 13)
    assert r["f1"] == pytest.approx(2 * (3 / 10) * (3 / 13) / ((3 / 10) + (3 / 13)))


def test_rates_zero_denominators_are_safe():
    r = metrics.rates(tp=0, tn=10, fp=0, fn=5)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


def test_roc_curve_perfect_separation_is_one():
    df = pd.DataFrame({"predicted_default_probability": [0.9, 0.8, 0.2, 0.1],
                       "default": [1, 1, 0, 0]})
    _, _, auc = metrics.roc_curve(df)
    assert auc == pytest.approx(1.0)


def test_roc_curve_random_is_in_unit_interval():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"predicted_default_probability": rng.random(200),
                       "default": rng.integers(0, 2, 200)})
    fpr, tpr, auc = metrics.roc_curve(df)
    assert 0.0 <= auc <= 1.0
    assert len(fpr) == len(tpr) and len(fpr) > 0


def test_roc_curve_empty_inputs():
    fpr, tpr, auc = metrics.roc_curve(pd.DataFrame())
    assert len(fpr) == 0 and auc == 0.0


def test_pr_curve_returns_arrays_and_auc_in_range():
    df = pd.DataFrame({"predicted_default_probability": [0.9, 0.7, 0.6, 0.3, 0.1],
                       "default": [1, 0, 1, 0, 0]})
    recall, precision, auc = metrics.pr_curve(df)
    assert 0.0 <= auc <= 1.0
    assert len(recall) == len(precision) and len(recall) > 0


def test_pr_curve_no_positives_is_empty():
    df = pd.DataFrame({"predicted_default_probability": [0.4, 0.3],
                       "default": [0, 0]})
    recall, precision, auc = metrics.pr_curve(df)
    assert len(recall) == 0 and auc == 0.0


def test_default_rate_by_group():
    df = pd.DataFrame({"occupancy_status": ["P", "P", "S", "I", "P"],
                       "default": [0, 1, 0, 1, 0]})
    g = metrics.default_rate_by_group(df, "occupancy_status")
    p_row = g[g["occupancy_status"] == "P"].iloc[0]
    assert p_row["loan_count"] == 3
    assert p_row["observed_defaults"] == 1
    assert p_row["observed_default_rate"] == pytest.approx(1 / 3)


def test_numeric_band_default_rate():
    df = pd.DataFrame({"upb": [50000, 150000, 350000, 700000, 120000],
                       "default": [1, 0, 0, 1, 1]})
    g = metrics.numeric_band_default_rate(
        df, "upb", bins=[-np.inf, 100000, 250000, 500000, np.inf],
        labels=["<$100k", "$100k-$250k", "$250k-$500k", "$500k+"])
    assert len(g) == 4
    under = g[g["band"] == "<$100k"].iloc[0]
    assert under["loan_count"] == 1 and under["observed_default_rate"] == 1.0


# --------------------------------------------------------------------------- #
# Data loaders: graceful handling of missing files
# --------------------------------------------------------------------------- #
def test_loaders_return_empty_when_files_missing(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    assert data_loader.load_model_metrics(missing) == {}
    assert data_loader.load_data_quality(missing) == {}
    assert data_loader.load_model_metrics_csv(missing).empty
    assert data_loader.load_validation_threshold_metrics(missing).empty
    assert data_loader.load_target_distribution(missing).empty
    assert data_loader.load_descriptive_statistics(missing).empty
    assert data_loader.load_balance_proxy(missing).empty


def test_load_test_predictions_missing_directory(tmp_path):
    assert data_loader.load_test_predictions(tmp_path / "no_dir").empty


def test_load_test_predictions_missing_parts(tmp_path):
    # Directory exists but has no part-*.csv files.
    (tmp_path / "_SUCCESS").write_text("")
    assert data_loader.load_test_predictions(tmp_path).empty


def test_load_test_predictions_concatenates_part_files(tmp_path):
    """Spark writes one header per part file; the loader must concat correctly."""
    header = "loan_sequence_number,default,prediction,predicted_default_probability,credit_score,dti,ltv,cltv,upb,occupancy_status,loan_purpose\n"
    part1 = header + "L1,0,0.0,0.10,700,20,80,80,150000,P,P\n" + "L2,1,1.0,0.60,650,40,90,90,200000,O,C\n"
    part2 = header + "L3,0,1.0,0.55,710,30,70,70,300000,P,P\n"
    (tmp_path / "part-00000.csv").write_text(part1)
    (tmp_path / "part-00001.csv").write_text(part2)
    # A CRC marker must be ignored by the part-*.csv glob.
    (tmp_path / ".part-00000.csv.crc").write_text("noise")

    df = data_loader.load_test_predictions(tmp_path)
    assert len(df) == 3
    assert list(df["loan_sequence_number"]) == ["L1", "L2", "L3"]
    assert df["default"].tolist() == [0, 1, 0]
    assert df["upb"].tolist() == [150000.0, 200000.0, 300000.0]
