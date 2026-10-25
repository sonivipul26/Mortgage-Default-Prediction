import json
from pyspark.sql import functions as F
from .config import TABLES

def explore(df):
    n = df.count(); duplicates = df.count() - df.dropDuplicates(["loan_sequence_number"]).count()
    missing = [(c, df.where(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == "")).count()) for c in df.columns]
    quality = [{"column": c, "missing_count": k, "missing_pct": round(100*k/n, 3) if n else None} for c,k in missing]
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "data_quality.json", "w", encoding="utf-8") as f: json.dump({"rows":n,"columns":len(df.columns),"duplicate_loans":duplicates,"missing":quality}, f, indent=2)
    df.describe().toPandas().to_csv(TABLES / "descriptive_statistics.csv", index=False)
    df.groupBy("default").count().toPandas().to_csv(TABLES / "target_distribution.csv", index=False)
    return {"rows":n, "columns":len(df.columns), "duplicate_loans":duplicates}
