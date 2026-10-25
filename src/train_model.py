from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as F
from .config import MODELS, SEED
from .data_preprocessing import preprocessing_stages

def split_by_loan(df):
    """One origin-level record per loan means randomSplit keeps loan IDs disjoint."""
    return df.randomSplit([0.70, 0.15, 0.15], seed=SEED)

def fit_model(train_df, use_class_weights=False):
    if train_df.select("default").distinct().count() < 2:
        raise ValueError("Training split has one class; provide a larger/more varied sample.")
    stages, nums, cats = preprocessing_stages(train_df)
    options = dict(labelCol="default", featuresCol="features", predictionCol="prediction", probabilityCol="probability", maxIter=100, regParam=0.05, elasticNetParam=0.0)
    fit_df = train_df
    if use_class_weights:
        counts = {row["default"]: row["count"] for row in train_df.groupBy("default").count().collect()}
        if not counts.get(0) or not counts.get(1): raise ValueError("Weighted fitting needs both target classes.")
        fit_df = train_df.withColumn("class_weight", F.when(F.col("default") == 1, F.lit(counts[0] / counts[1])).otherwise(F.lit(1.0)))
        options["weightCol"] = "class_weight"
    model = Pipeline(stages=stages + [LogisticRegression(**options)]).fit(fit_df)
    return model, {"numeric": nums, "categorical": cats, "class_weighted": use_class_weights}

def save_model(model):
    path = MODELS / "mortgage_logistic_regression"
    model.write().overwrite().save(str(path))
    return str(path)
