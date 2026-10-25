from pyspark.sql import functions as F

def validate_observations(df):
    required = {"loan_sequence_number", "default"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    invalid = df.where(~F.col("default").isin(0, 1)).count()
    if invalid: raise ValueError(f"default must be binary; found {invalid} invalid rows")
    if df.select("loan_sequence_number").where(F.col("loan_sequence_number").isNull()).count(): raise ValueError("Null loan identifiers")
    return True
