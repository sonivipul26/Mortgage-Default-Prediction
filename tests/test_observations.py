from pyspark.sql.types import DoubleType
from src.data_loader import ORIG_COLS, PERF_COLS, build_observations
from src.data_preprocessing import NUMERIC

def test_build_observations_casts_all_configured_numeric_predictors(spark):
    """Regression test for string CSV columns reaching Spark's numeric Imputer."""
    orig_values = {column: "" for column in ORIG_COLS}
    orig_values.update({"loan_sequence_number": "loan-1", "credit_score": "720", "dti": "35", "ltv": "80", "cltv": "85", "upb": "250000"})
    perf_values = {column: "" for column in PERF_COLS}
    perf_values.update({"loan_sequence_number": "loan-1", "monthly_reporting_period": "202201", "current_loan_delinquency_status": "00"})
    orig = spark.createDataFrame([tuple(orig_values[column] for column in ORIG_COLS)], ORIG_COLS)
    perf = spark.createDataFrame([tuple(perf_values[column] for column in PERF_COLS)], PERF_COLS)

    observations = build_observations(orig, perf, observation_months=1)

    for column in NUMERIC:
        assert isinstance(observations.schema[column].dataType, DoubleType)
