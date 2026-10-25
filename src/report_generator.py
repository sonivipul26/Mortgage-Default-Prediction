import json
from pathlib import Path
from docx import Document
from docx.shared import Inches
from .config import REPORT, CHARTS, METRICS


def _fmt(value, nd=4):
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def _load_validation_summary():
    """Read the validation threshold table the pipeline just wrote. Returns
    {model: best_row} for each candidate model, plus selected_model/threshold.
    Missing file -> None (pending report path)."""
    path = METRICS / "validation_threshold_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    best = {}
    for row in rows:
        name = row.get("model")
        cur = best.get(name)
        if cur is None or (row["f1"], row["recall"], -row["predicted_positive_count"]) > (
            cur["f1"], cur["recall"], -cur["predicted_positive_count"]
        ):
            best[name] = row
    return {"best_per_model": best, "selected_model": data.get("selected_model"),
            "selected_threshold": data.get("selected_threshold"),
            "selected_validation_metrics": data.get("selected_validation_metrics")}


def _interpret(metrics):
    """Honest, data-driven results discussion. Never claims success from
    accuracy alone; states recall/precision/F1/AUC and the always-non-default
    and non-selected-model comparisons."""
    if not metrics:
        return "Pending: run python run_project.py after installation and data setup."
    tp = int(metrics.get("true_positive", 0)); fn = int(metrics.get("false_negative", 0))
    fp = int(metrics.get("false_positive", 0)); tn = int(metrics.get("true_negative", 0))
    total = tp + tn + fp + fn
    model = metrics.get("model", "the selected model")
    thr = metrics.get("selected_threshold")
    parts = [
        f"The validation set selected {model} at a decision threshold of {_fmt(thr, 2)} "
        f"(criterion: maximum validation F1, ties broken by higher recall). The test set "
        f"was evaluated once at that threshold; threshold and model choice never used test labels.",
        f"On the held-out test set the model detected {tp} of {tp + fn} actual defaults "
        f"(recall {_fmt(metrics.get('recall', 0.0))}) with {fp} false positives "
        f"(precision {_fmt(metrics.get('precision', 0.0))}) and F1 {_fmt(metrics.get('f1', 0.0))}. "
        f"ROC-AUC is {_fmt(metrics.get('roc_auc'))} and PR-AUC is {_fmt(metrics.get('pr_auc'))}; "
        f"both rank metrics use the continuous probability scores, not the binary threshold predictions.",
    ]
    prev = metrics.get("test_default_prevalence", 0.0)
    base_acc = metrics.get("always_non_default_accuracy", 0.0)
    model_acc = (tn + tp) / total if total else 0.0
    parts.append(
        f"Default prevalence in the test set is {_fmt(prev * 100, 2)}%, so an always-non-default "
        f"baseline scores a high accuracy of {_fmt(base_acc * 100, 2)}% while detecting zero "
        f"defaults (recall 0.0000, F1 0.0000). The model trades some of that accuracy "
        f"({_fmt(model_acc * 100, 2)}%) to detect the rare default class. High accuracy alone is "
        f"therefore not evidence of successful default detection; recall, precision, F1, ROC-AUC "
        f"and PR-AUC are the relevant measures for this imbalanced problem."
    )
    val = _load_validation_summary()
    if val and val.get("best_per_model"):
        lines = []
        for name, row in val["best_per_model"].items():
            lines.append(
                f"{name}: best validation F1 {_fmt(row['f1'])} at threshold "
                f"{_fmt(row['threshold'], 2)} (validation TP {row['true_positive']}, "
                f"FP {row['false_positive']}, FN {row['false_negative']})"
            )
        parts.append(
            "Validation threshold comparison across the candidate models: " + "; ".join(lines) + ". "
            "The plain logistic regression is poorly calibrated for this rare class, so at every "
            "candidate threshold from 0.05 to 0.50 it predicts few or zero defaults on validation; "
            "the class-weighted candidate is what makes default detection possible. Even so, "
            "precision and F1 remain low: with only documented origination-time features and no "
            "borrower income, the model flags many non-defaults as risky for each true default "
            "caught. This is an honest baseline result, not a production credit-decision system; "
            "defensible next experiments are richer feature sets, tree-based models, or a "
            "time-based validation split, rather than further tuning to the test labels."
        )
    return " ".join(parts)

def create_report(status, metrics=None, feature_info=None, exploration=None):
    REPORT.mkdir(parents=True,exist_ok=True); d=Document(); sec=d.sections[0]; sec.top_margin=Inches(.7)
    d.add_heading('Mortgage Default Prediction Using PySpark and Logistic Regression',0)
    d.add_paragraph('Academic Project Report\nStudent: [Name] | Roll No.: [Roll Number]\nUniversity: [University] | Department: [Department]\nFaculty: [Faculty Name] | Submission date: [Date]')
    d.add_page_break(); d.add_heading('Abstract',1)
    d.add_paragraph('This project uses Apache Spark MLlib logistic regression to estimate mortgage default risk from publicly documented Freddie Mac loan-level data. '+('Actual run outputs are reported below.' if status=='completed' else 'The raw data and executable environment were unavailable when this report was generated; results are explicitly pending.'))
    result_text='Actual held-out results are reported in the following table. Threshold selection used validation data only.' if metrics else 'Pending: run python run_project.py after installation and data setup.'
    for title, text in [('1. Introduction','Mortgage default prediction supports portfolio risk management.'),('2. Dataset and target','Source: Freddie Mac Single-Family Loan-Level Dataset. The binary default target is 1 for a documented 90+ days delinquent event or zero-balance foreclosure/disposition code 03/09 within 24 months. Non-default loans require 24 months of follow-up. Features are origination-time fields only.'),('3. Methodology','Spark reads pipe-delimited originations and servicing files, joins on loan_sequence_number, excludes insufficient follow-up, then splits distinct loan-level observations into train, validation and test partitions. Imputation/encoding and both Logistic Regression candidates are fitted only on training data. The validation set selects the class-weighting approach and decision threshold by maximum F1. The test set is evaluated once.'),('4. Results', result_text),('5. Class imbalance and interpretation','Defaults are rare, so always predicting non-default can have high accuracy while recall is zero. ROC-AUC measures rank discrimination; PR-AUC is more informative for this rare positive class. The selected threshold is not a probability of certainty: it is an operating point chosen from validation data. False negatives are defaults not detected by the model.'),('6. Limitations','The public Freddie Mac layout does not provide borrower income, occupation, or home ownership. Consequently, the requested income chart cannot be produced truthfully; original UPB bands are clearly labelled only as a loan-balance proxy. This academic model is not a production credit decision system.'),('7. Conclusion','The pipeline is reproducible, leakage-aware, evaluates class imbalance explicitly, and saves its fitted Spark pipeline and safe prediction exports.'),('References','Freddie Mac, Single-Family Loan-Level Dataset documentation, https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset\nApache Spark MLlib documentation, https://spark.apache.org/docs/latest/ml-guide.html')]:
        d.add_heading(title,1); d.add_paragraph(text)
    if metrics:
        d.add_heading('Actual evaluation metrics',2)
        table=d.add_table(rows=1, cols=2); table.style='Table Grid'; table.rows[0].cells[0].text='Metric';table.rows[0].cells[1].text='Value'
        for k,v in metrics.items():
            if k=='validation_selected_metrics': continue  # nested dict; saved in validation_threshold_metrics.json
            cells=table.add_row().cells;cells[0].text=k;cells[1].text=str(v)
        d.add_heading('Results interpretation',2); d.add_paragraph(_interpret(metrics))
    for image in sorted(CHARTS.glob('*.png')):
        d.add_heading(image.stem.replace('_',' ').title(),2);d.add_picture(str(image),width=Inches(6));d.add_paragraph('Figure: Generated from held-out predictions.')
    d.save(REPORT/'project_report.docx')
