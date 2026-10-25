from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler

NUMERIC = ["credit_score", "dti", "ltv", "cltv", "upb"]
CATEGORICAL = ["occupancy_status", "loan_purpose", "property_type", "channel"]

def available_features(df):
    return [c for c in NUMERIC if c in df.columns], [c for c in CATEGORICAL if c in df.columns]

def preprocessing_stages(df):
    nums, cats = available_features(df)
    stages = []
    if nums: stages.append(Imputer(strategy="median", inputCols=nums, outputCols=[f"{c}_imp" for c in nums]))
    indexed = []
    for c in cats:
        idx=f"{c}_idx"; indexed.append(idx); stages.append(StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep"))
    encoded=[]
    if indexed:
        encoded=[f"{c}_oh" for c in cats]; stages.append(OneHotEncoder(inputCols=indexed, outputCols=encoded, handleInvalid="keep"))
    inputs=[f"{c}_imp" for c in nums]+encoded
    if not inputs: raise ValueError("No supported model features exist in dataset")
    stages.append(VectorAssembler(inputCols=inputs, outputCol="features", handleInvalid="keep"))
    return stages, nums, cats
