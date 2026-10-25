import json
import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from .config import METRICS

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

def metric_row(tp, tn, fp, fn, threshold):
    total=tp+tn+fp+fn; precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    return {"threshold":threshold,"accuracy":(tp+tn)/total if total else 0.0,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,"true_positive":tp,"true_negative":tn,"false_positive":fp,"false_negative":fn,"predicted_positive_count":tp+fp,"records":total}

def add_probability(predictions):
    from pyspark.ml.functions import vector_to_array
    return predictions.withColumn("predicted_default_probability", vector_to_array("probability")[1])

def threshold_metrics(predictions, thresholds=THRESHOLDS):
    """Collect (default, probability) once and compute the confusion matrix for
    every threshold locally. This is identical to a per-threshold Spark
    aggregation but scans the predictions a single time instead of once per
    threshold (the old design triggered one full pipeline transform scan per
    threshold and was impractically slow)."""
    pdf = predictions.select("default", "predicted_default_probability").toPandas()
    y = pdf["default"].to_numpy()
    p = pdf["predicted_default_probability"].to_numpy()
    rows = []
    for threshold in thresholds:
        predicted_positive = p >= threshold
        tp = int(((y == 1) & predicted_positive).sum())
        tn = int(((y == 0) & ~predicted_positive).sum())
        fp = int(((y == 0) & predicted_positive).sum())
        fn = int(((y == 1) & ~predicted_positive).sum())
        rows.append(metric_row(tp, tn, fp, fn, float(threshold)))
    return rows

def select_threshold(rows):
    return max(rows,key=lambda row:(row["f1"],row["recall"],-row["predicted_positive_count"]))

def roc_pr_auc(predictions):
    evaluator=BinaryClassificationEvaluator(labelCol="default",rawPredictionCol="rawPrediction")
    return evaluator.setMetricName("areaUnderROC").evaluate(predictions), evaluator.setMetricName("areaUnderPR").evaluate(predictions)

def save_validation_results(rows, selected_model, selected):
    METRICS.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(METRICS/"validation_threshold_metrics.csv",index=False)
    with open(METRICS/"validation_threshold_metrics.json","w",encoding="utf-8") as file: json.dump({"selection_criterion":"maximum validation F1; ties: higher recall, then fewer predicted positives","selected_model":selected_model,"selected_threshold":selected["threshold"],"selected_validation_metrics":selected,"rows":rows},file,indent=2)

def evaluate_test(predictions, train_count, validation_count, selected, model_name):
    METRICS.mkdir(parents=True,exist_ok=True); result=threshold_metrics(predictions,[selected["threshold"]])[0]; total=result["records"]; positives=result["true_positive"]+result["false_negative"]; roc,pr=roc_pr_auc(predictions)
    result.update({"model":model_name,"selected_threshold":selected["threshold"],"training_records":train_count,"validation_records":validation_count,"test_records":total,"test_default_prevalence":positives/total if total else 0.0,"roc_auc":roc,"pr_auc":pr,"always_non_default_accuracy":(total-positives)/total if total else 0.0,"always_non_default_precision":0.0,"always_non_default_recall":0.0,"always_non_default_f1":0.0,"threshold_selection":"maximum validation F1; ties: higher recall, then fewer predicted positives","validation_selected_metrics":selected})
    with open(METRICS/"model_metrics.json","w",encoding="utf-8") as file: json.dump(result,file,indent=2)
    pd.DataFrame([result]).drop(columns=["validation_selected_metrics"]).to_csv(METRICS/"model_metrics.csv",index=False)
    return result
