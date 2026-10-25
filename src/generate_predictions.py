from pyspark.sql import functions as F
from .config import PREDICTIONS

def save_predictions(model, test_df, predictions=None):
    pred = predictions if predictions is not None else model.transform(test_df)
    keep=[c for c in ["loan_sequence_number","default","prediction","predicted_default_probability","credit_score","dti","ltv","cltv","upb","occupancy_status","loan_purpose"] if c in pred.columns]
    safe=pred.select(*keep)
    safe.write.mode("overwrite").option("header",True).csv(str(PREDICTIONS / "test_predictions_csv"))
    return pred
