from src.evaluate_model import metric_row, select_threshold

def test_threshold_metrics_confusion_matrix_is_consistent():
    row=metric_row(3, 90, 7, 10, 0.20)
    assert row["records"] == row["true_positive"] + row["true_negative"] + row["false_positive"] + row["false_negative"]
    assert row["predicted_positive_count"] == row["true_positive"] + row["false_positive"]

def test_threshold_selection_maximises_validation_f1():
    selected=select_threshold([metric_row(1,95,1,9,0.10),metric_row(5,85,11,5,0.20)])
    assert selected["threshold"] == 0.20
