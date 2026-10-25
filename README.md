# Mortgage Default Prediction Using PySpark and Logistic Regression

An executable, leakage-aware academic project. It uses **Spark MLlib**, not scikit-learn, for training. No dashboard functionality is included.

## What is implemented

- Freddie Mac public originations/performance loader, documented field names, loan-ID join, and a 24-month default definition.
- Spark EDA and data-quality exports; training-only median imputation and safe categorical encoding; MLlib Logistic Regression; model persistence; held-out predictions, metrics, charts, Word report, presentation notes, and pytest tests.
- The project never fabricates data or performance metrics.

## Dataset and required manual download

The preferred source is [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset). Its download site requires acceptance of its terms/registration in some cases. Download matching **yearly sample** origination and servicing files, retain their pipe-delimited contents, and put them here:

```text
data/raw/sample_orig_YYYY.txt
data/raw/sample_svcg_YYYY.txt
```

This project also accepts `sample_perf_YYYY.txt` as a local alias for the servicing file; `sample_perf_2022.txt` is supported. The 2022 sample is headerless and pipe-delimited: its origination file has 31 fields and its performance file has 35. The loader checks those counts before Spark reads a file, preventing a mismatched format from shifting columns silently. The documented `loan_sequence_number` joins the files. The target is `default=1` if servicing status is 3+ (90+ DPD) or zero-balance code 03/09 appears **within** the first 24 monthly performance observations. `default=0` requires at least 24 months of follow-up without such an event. Insufficiently observed non-events are excluded. This uses only origination-time predictors, avoiding performance-data leakage.

Freddie Mac's public layout does **not** provide borrower income, occupation, or home ownership. This project does not claim those features exist. `income_risk_summary.csv` is only generated as a clearly labelled original-UPB proxy table when appropriate; it is not income analysis.

Review and comply with Freddie Mac data terms before use. Do not commit raw loan files or sensitive exports.

## Windows setup

Install Python 3.10 or 3.11 (add it to PATH) and a 64-bit JDK 8 or 11. This project is pinned to PySpark 3.5.1, which supports Python 3.8–3.11 and Java 8/11/17.

```powershell
cd mortgage-default-prediction
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_project.py
pytest -q
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`. `python run_project.py` fails clearly, creates a pending report, and prints setup instructions if raw files are absent.

## Outputs

- `models/mortgage_logistic_regression/`: native saved Spark PipelineModel.
- `outputs/metrics/`: JSON/CSV accuracy, ROC-AUC, precision, recall, F1, confusion matrix counts, prevalence, and record counts.
- `outputs/metrics/validation_threshold_metrics.csv`: validation-only comparison of baseline and class-weighted Logistic Regression across thresholds 0.05–0.50. The selected configuration maximises validation F1 (ties favour recall, then fewer predicted positives).
- `outputs/predictions/`: safe test prediction CSV directory (identifier, outcome, probability, permitted feature fields).
- `outputs/charts/` and `outputs/tables/`: charts, quality JSON, descriptive statistics, and risk summary.
- `report/project_report.docx`: generated after the run, including real charts and metrics.

## Individual stages

The modules are importable: `data_loader.load_freddie_mac`, `data_exploration.explore`, `train_model.train`, `evaluate_model.evaluate`, `create_visualizations.create_charts`, and `report_generator.create_report`. `run_project.py` is the supported ordered execution path.

## Interpretation and limitations

Accuracy alone is weak for rare events; inspect ROC-AUC, PR-AUC (Spark's area-under-precision-recall), recall, precision, F1, and confusion-matrix counts. The train/validation/test split is loan-level and uses validation—not test labels—to select both class weighting and the operating threshold. The test set is then evaluated once. Results are data- and period-dependent. The project intentionally omits unavailable income/occupation/home-ownership features, uses a defined horizon, and should not be used for production credit decisions without governance, bias validation, calibration, and independent review.

## Interactive Dashboard

A Streamlit dashboard reads the pipeline's generated outputs (metrics, predictions, tables) and presents them in a professional dark fintech interface. **No model retraining happens when the dashboard starts.**

### Dashboard sections

| Section | Content |
|---|---|
| Executive Overview | KPI cards, default vs non-default chart, risk distribution |
| Loan Risk Analysis | Default rates by occupancy/purpose/LTV/DTI, UPB-band proxy, filterable prediction table |
| Model Performance | Confusion matrix, ROC curve, PR curve, threshold comparison |
| Prediction Explorer | Per-loan detail, risk gauge, top-N high-risk table |
| Project Information | Dataset source, pipeline diagram, methodology, disclaimer |

### Run the dashboard (Windows PowerShell)

The pipeline must be run at least once before starting the dashboard:

```powershell
cd C:\Users\vipul\Desktop\Cluster\mortgage-default-prediction
.\.venv\Scripts\Activate.ps1

# 1. Run the ML pipeline first (if not already done)
python run_project.py

# 2. Start the dashboard
streamlit run dashboard/app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).
Open it in your browser. Use Ctrl+C in PowerShell to stop.

### Dashboard-specific dependencies

Streamlit and Plotly are already listed in `requirements.txt`. Install with:

```powershell
pip install -r requirements.txt
```

### Dashboard tests

```powershell
python -m pytest tests/test_dashboard_data.py -q
```

The dashboard tests use synthetic fixtures and do not require a Spark session.
They verify data loaders, metric calculations, and graceful handling of missing files.

## Exact execution and viewing commands

```powershell
cd C:\Users\vipul\Desktop\Cluster\mortgage-default-prediction
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run_project.py
Invoke-Item .\outputs\metrics\model_metrics.json
Invoke-Item .\report\project_report.docx
```

## References

- Freddie Mac: Single-Family Loan-Level Dataset and file documentation.
- Apache Spark MLlib: Logistic Regression and Pipeline documentation.
