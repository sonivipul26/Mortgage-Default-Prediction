# Presentation and Viva Notes

## Two-minute introduction
Mortgage default prediction estimates the likelihood that a loan will enter serious delinquency or foreclosure. This project uses PySpark because mortgage servicing data can be large, and uses Spark MLlib Logistic Regression because it is interpretable and produces probabilities. It joins Freddie Mac origination and performance records by the documented loan sequence number. The label is a 90+ day delinquency or documented foreclosure/disposition event within 24 months; loans with inadequate follow-up are excluded.

## Key talking points
- PySpark distributes transformations and supports a repeatable Spark ML Pipeline.
- Median imputation and encoders are fitted on training data only, preventing leakage.
- Categorical values use StringIndexer and OneHotEncoder with safe unseen-value handling.
- Accuracy can be misleading for rare defaults; ROC-AUC measures ranking, while precision, recall and F1 expose positive-class performance.
- Predicted probability is a model estimate; observed default rate is historical outcome frequency.
- Freddie Mac public sample data does not include borrower income, occupation, or home ownership. The project does not invent them; any UPB grouping is explicitly a loan-balance proxy.

## Architecture and current results
Raw Freddie Mac origination and monthly servicing files are validated, joined by loan sequence number, and converted into one origin-time feature row per eligible loan. The 24-month target is made only from performance records; performance fields never enter the model features. Loans are split into train, validation and test sets before preprocessing. The training set fits median imputation, categorical encoders and two Logistic Regression candidates: baseline and inverse-frequency weighted. Validation chooses the model and threshold that maximise F1; test is used once for final metrics. Read the actual completed metrics from `outputs/metrics/model_metrics.json` and the threshold comparison from `outputs/metrics/validation_threshold_metrics.csv`.

ROC-AUC measures whether defaults tend to receive higher scores than non-defaults. PR-AUC focuses on precision/recall and is more useful for rare defaults. A 0.50 threshold may have excellent accuracy while detecting almost no defaults, so threshold selection and false-negative counts must be discussed.

## Headline results (latest run)
The full dataset is 45,697 eligible loans with 856 defaults (1.87% prevalence). Splits are train 32,206 / validation 6,813 / test 6,678. Validation selected the class-weighted logistic regression at threshold 0.50 (maximum validation F1; ties favour recall).

- Test set: detected 96 of 131 defaults. Recall 0.7328, precision 0.0432, F1 0.0817, accuracy 0.6767.
- ROC-AUC 0.7711 (rank quality of the continuous scores); PR-AUC 0.0572.
- Always-non-default baseline: accuracy 0.9804, but recall 0.0000 and F1 0.0000 (it detects nothing; high accuracy is misleading here).
- The plain (non-weighted) logistic regression is poorly calibrated for this rare class: its predicted probabilities are almost all below 0.05, so at every candidate threshold 0.05–0.50 it predicts zero defaults on validation. Class weighting is what makes default detection possible at all.

Honest reading: the model genuinely detects defaults (73% recall, up from 0%), but precision and F1 are low because origination-only features are weak predictors and defaults are rare. This is an academic baseline, not a production credit-decision system. Read the exact numbers from `outputs/metrics/model_metrics.json` and the full threshold comparison from `outputs/metrics/validation_threshold_metrics.csv`.

## Likely viva questions
1. **Why Logistic Regression?** It is interpretable, fast, and estimates class probabilities.
2. **Why PySpark?** It handles datasets too large for a single machine and has MLlib pipelines.
3. **What is the target?** A 90+ DPD or qualifying foreclosure/disposition event within 24 months.
4. **Why exclude short follow-up loans?** Their future outcome is unknown; calling them non-default biases labels.
5. **What is leakage?** Using future performance or test-set information to build a prediction model.
6. **How is leakage prevented?** Only origination features are used and preprocessing is fitted on training data.
7. **Why median imputation?** It is robust to extreme numerical values.
8. **Why one-hot encoding?** It represents categories without imposing artificial order.
9. **What is ROC-AUC?** The probability that a random default is ranked above a random non-default.
10. **What is precision?** Of predicted defaults, the fraction that really defaulted.
11. **What is recall?** Of real defaults, the fraction found by the model.
12. **Why can accuracy mislead?** Predicting all loans non-default can be accurate when defaults are rare.
13. **What does regularization do?** It controls coefficient size and reduces overfitting.
14. **Why save the pipeline?** Inference must use exactly the training transformations.
15. **Main limitation?** Public sample fields omit several borrower attributes and results depend on the chosen observation window.
16. **Why is the test set not used to select a threshold?** Doing so leaks test-label information and gives overly optimistic reported performance.
17. **What is the always-non-default baseline?** It predicts no defaults; its accuracy equals the non-default prevalence but its recall and F1 are zero.

## Challenges and solutions
Large files: select a manageable quarterly sample first. Missing fields: validate schema and use only documented available fields. Class imbalance: report ROC-AUC/precision/recall, not accuracy alone.
