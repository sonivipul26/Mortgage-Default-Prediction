from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from .config import CHARTS, TABLES, FIGURES

def _save(name):
    for folder in (CHARTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True); plt.savefig(folder / name, dpi=180, bbox_inches="tight")
    plt.close()

def create_charts(pred, validation_rows=None, metrics=None):
    sns.set_theme(style="whitegrid")
    pred.groupBy("default").count().toPandas().plot.bar(x="default",y="count",legend=False,title="Default versus non-default"); _save("default_distribution.png")
    if "credit_score" in pred.columns:
        p=pred.select("credit_score","predicted_default_probability").dropna().sample(False,0.1,42).toPandas()
        if not p.empty: sns.scatterplot(data=p,x="credit_score",y="predicted_default_probability",alpha=.35); plt.title("Credit score versus predicted default probability"); _save("credit_score_probability.png")
    # Freddie Mac lacks borrower income. UPB bands are a disclosed loan-balance proxy, not income.
    if "upb" in pred.columns:
        x=pred.withColumn("balance_band",F.when(F.col("upb")<100000,"<$100k").when(F.col("upb")<250000,"$100k-$250k").when(F.col("upb")<500000,"$250k-$500k").otherwise("$500k+"))
        summary=x.groupBy("balance_band").agg(F.count("*").alias("borrower_count"),F.avg("predicted_default_probability").alias("average_predicted_default_probability"),F.avg("default").alias("observed_default_rate"),F.avg("credit_score").alias("average_credit_score")).toPandas()
        summary.to_csv(TABLES / "income_risk_summary.csv",index=False) # required filename; content labelled proxy
        summary.plot.bar(x="balance_band",y="average_predicted_default_probability",legend=False,title="Predicted default probability by original UPB band (not income)"); _save("predicted_probability_by_balance_proxy.png")
        summary.plot.bar(x="balance_band",y="observed_default_rate",legend=False,title="Observed default rate by original UPB band (not income)"); _save("observed_default_by_balance_proxy.png")
    p=pred.select("predicted_default_probability").toPandas(); sns.histplot(p["predicted_default_probability"],bins=30); plt.title("Predicted default probability distribution"); _save("predicted_probability_distribution.png")
    labelled=pred.select("default","predicted_default_probability").toPandas()
    sns.histplot(data=labelled,x="predicted_default_probability",hue="default",bins=30,stat="density",common_norm=False); plt.title("Probability distribution: defaults versus non-defaults"); _save("probability_by_actual_class.png")
    if validation_rows:
        frame=pd.DataFrame(validation_rows); selected=metrics["model"] if metrics else frame["model"].iloc[0]; frame=frame[frame["model"]==selected]
        frame.plot(x="threshold",y=["precision","recall","f1"],marker="o",title="Validation precision, recall and F1 by threshold"); plt.ylabel("Metric"); _save("threshold_metrics.png")
    # Curves use the small held-out prediction table, never the full raw data.
    curve=labelled.sort_values("predicted_default_probability",ascending=False); positives=curve.default.sum(); negatives=len(curve)-positives
    curve["tp"]=curve.default.cumsum(); curve["fp"]=(1-curve.default).cumsum()
    if positives and negatives:
        plt.plot(curve.fp/negatives,curve.tp/positives); plt.plot([0,1],[0,1],"--",color="grey"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve (held-out test set)"); _save("roc_curve.png")
        precision=curve.tp/(curve.tp+curve.fp); plt.plot(curve.tp/positives,precision); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall curve (held-out test set)"); _save("precision_recall_curve.png")
    if metrics:
        matrix=[[metrics["true_negative"],metrics["false_positive"]],[metrics["false_negative"],metrics["true_positive"]]]
        sns.heatmap(matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Predicted non-default","Predicted default"],yticklabels=["Actual non-default","Actual default"]); plt.title(f"Confusion matrix at threshold {metrics['selected_threshold']:.2f}"); _save("confusion_matrix.png")
