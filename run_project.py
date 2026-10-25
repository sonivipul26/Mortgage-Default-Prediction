"""One-command pipeline. Never reports model results unless every training stage succeeds."""
import logging, sys
from src.config import ensure_directories
from src.spark_session import get_spark
from src.data_loader import load_freddie_mac, build_observations
from src.data_validation import validate_observations
from src.data_exploration import explore
from src.train_model import split_by_loan, fit_model, save_model
from src.generate_predictions import save_predictions
from src.evaluate_model import add_probability, threshold_metrics, select_threshold, save_validation_results, evaluate_test
from src.create_visualizations import create_charts
from src.feature_engineering import feature_documentation
from src.report_generator import create_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
def main():
    ensure_directories()
    try:
        spark=get_spark(); orig, perf=load_freddie_mac(spark); observations=build_observations(orig,perf).cache()
        validate_observations(observations); eda=explore(observations)  # EDA's many per-column counts reuse the cached observations
        train_df, validation_df, test_df=split_by_loan(observations)
        # Cache partitions so both model fits and every downstream transform/evaluation reuse
        # materialized data instead of re-running the multi-million-row join on each action.
        train_df=train_df.cache(); validation_df=validation_df.cache(); test_df=test_df.cache()
        # Both candidates fit preprocessing only on train; validation labels choose model/threshold.
        candidates=[]
        for name, weighted in (("baseline_logistic_regression",False),("class_weighted_logistic_regression",True)):
            model, features=fit_model(train_df,weighted); validation_predictions=add_probability(model.transform(validation_df))
            rows=[dict(model=name,**row) for row in threshold_metrics(validation_predictions)]
            candidates.append((name,model,features,select_threshold(rows),rows))
        name,model,features,selected,all_rows=max(candidates,key=lambda item:(item[3]["f1"],item[3]["recall"],-item[3]["predicted_positive_count"]))
        all_rows=[row for item in candidates for row in item[4]]
        save_validation_results(all_rows,name,selected); features["model_path"]=save_model(model)
        pred=add_probability(model.transform(test_df)).cache(); save_predictions(model,test_df, predictions=pred)  # charts/eval reuse cached predictions
        metrics=evaluate_test(pred,train_df.count(),validation_df.count(),selected,name); create_charts(pred,all_rows,metrics)
        create_report('completed',metrics,features,eda); spark.stop()
        print('Completed. See outputs/, models/mortgage_logistic_regression/, and report/project_report.docx')
    except FileNotFoundError as e:
        create_report('pending')
        print(f'PIPELINE NOT RUN: {e}\nInstall Python 3.10/3.11, pip install -r requirements.txt, then put matching Freddie Mac sample_orig_YYYY.txt and sample_svcg_YYYY.txt (or sample_perf_YYYY.txt) files in data/raw/.')
        return 2
    except Exception as e:
        logging.exception('Pipeline failed'); print(f'PIPELINE FAILED: {e}'); return 1
if __name__=='__main__': sys.exit(main())
