"""Validated loader for the pre-July-2026 Freddie Mac yearly sample files."""
import csv
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, StringType
from .config import DATA_RAW, ORIGINATION_GLOB, PERFORMANCE_GLOBS, TARGET_MONTHS

# Public Freddie Mac files are pipe-delimited and have no header. Fields used are documented names.
ORIG_COLS = ["credit_score", "first_payment_date", "first_time_homebuyer_flag", "maturity_date", "msa", "mortgage_insurance_pct", "units", "occupancy_status", "cltv", "dti", "upb", "ltv", "interest_rate", "channel", "prepayment_penalty", "product_type", "property_state", "property_type", "postal_code", "loan_sequence_number", "loan_purpose", "original_loan_term", "number_of_borrowers", "seller_name", "super_conforming_flag", "pre_harp_loan_sequence_number", "program_indicator", "harp_indicator", "property_valuation_method", "interest_only_indicator", "mortgage_insurance_cancellation_indicator"]
# The 2022 yearly sample has 35 performance fields. The model uses only the
# documented prefix through zero_balance_code; trailing fields are retained so
# that records are never shifted or truncated by an incorrect schema.
PERF_COLS = ["loan_sequence_number", "monthly_reporting_period", "current_actual_upb", "current_loan_delinquency_status", "loan_age", "remaining_months_to_legal_maturity", "repurchase_flag", "modification_flag", "zero_balance_code", "zero_balance_effective_date", "current_interest_rate", "current_deferred_upb", "due_date_of_last_paid_installment", "mi_recoveries", "net_sales_proceeds", "non_mi_recoveries", "expenses", "legal_costs", "maintenance_and_preservation_costs", "taxes_and_insurance", "miscellaneous_expenses", "actual_loss_calculation", "modification_cost", "step_modification_flag", "deferred_payment_modification", "estimated_loan_to_value", "zero_balance_removal_upb", "delinquent_accrued_interest", "delinquency_due_to_disaster", "borrower_assistance_status", "current_month_modification_cost", "interest_bearing_upb", "performance_extension_33", "performance_extension_34", "performance_extension_35"]

def _read(spark, paths, cols):
    schema = StructType([StructField(column, StringType(), True) for column in cols])
    return (spark.read.option("sep", "|").option("header", False)
            .option("mode", "FAILFAST").schema(schema).csv([str(p) for p in paths]))

def _field_count(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        first = next(csv.reader(handle, delimiter="|"), None)
    return len(first) if first else 0

def _validate_layout(paths, expected, file_kind):
    invalid = [(path.name, _field_count(path)) for path in paths if _field_count(path) != expected]
    if invalid:
        raise ValueError(f"Unsupported {file_kind} layout (expected {expected} pipe-delimited fields): {invalid}")

def discover_raw_files():
    orig = sorted(DATA_RAW.glob(ORIGINATION_GLOB))
    perf = sorted({path for pattern in PERFORMANCE_GLOBS for path in DATA_RAW.glob(pattern)})
    return orig, perf

def load_freddie_mac(spark):
    orig, perf = discover_raw_files()
    if not orig or not perf:
        raise FileNotFoundError("Freddie Mac yearly sample files are missing. Expected sample_orig_YYYY.txt plus sample_svcg_YYYY.txt or sample_perf_YYYY.txt in data/raw/.")
    _validate_layout(orig, len(ORIG_COLS), "origination")
    _validate_layout(perf, len(PERF_COLS), "performance")
    return _read(spark, orig, ORIG_COLS), _read(spark, perf, PERF_COLS)

def build_observations(orig, perf, observation_months=TARGET_MONTHS):
    """One origin-level row per loan; default is a qualifying event *within* horizon."""
    p = (perf.withColumn("report_date", F.to_date("monthly_reporting_period", "yyyyMM"))
         .withColumn("delinq_num", F.coalesce(F.col("current_loan_delinquency_status").cast("int"), F.lit(0))))
    if p.where(F.col("report_date").isNull()).limit(1).count():
        raise ValueError("Unparseable monthly_reporting_period; expected yyyyMM, e.g. 202202.")
    horizon = p.groupBy("loan_sequence_number").agg(F.min("report_date").alias("first_report"), F.max("report_date").alias("last_report"))
    in_horizon = F.months_between(F.col("report_date"), F.col("first_report")) <= observation_months - 1
    events = (p.join(horizon.select("loan_sequence_number", "first_report"), "loan_sequence_number")
              .groupBy("loan_sequence_number")
              .agg(F.max(F.when(in_horizon & ((F.col("delinq_num") >= 3) | F.col("zero_balance_code").isin("03", "09")), 1).otherwise(0)).alias("default")))
    joined = orig.join(horizon, "loan_sequence_number").join(events, "loan_sequence_number")
    # loans need a full follow-up window unless they default; this avoids labelling censored loans non-default.
    eligible = (F.months_between("last_report", "first_report") >= observation_months - 1) | (F.col("default") == 1)
    return (joined.where(eligible).dropDuplicates(["loan_sequence_number"])
            .withColumn("credit_score", F.col("credit_score").cast("double"))
            .withColumn("dti", F.regexp_replace("dti", "%", "").cast("double"))
            .withColumn("ltv", F.col("ltv").cast("double"))
            .withColumn("cltv", F.col("cltv").cast("double"))
            .withColumn("upb", F.col("upb").cast("double")))
