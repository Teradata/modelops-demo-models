from sklearn import metrics
from teradataml import (
    ClassificationEvaluator,
    DataFrame,
    configure,
    copy_to_sql
)
from teradataml.analytics.valib import *
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import os
import joblib
import json
import uuid
import numpy as np
import pandas as pd


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    model = joblib.load(f"{context.artifact_input_path}/model.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    test_df = DataFrame.from_query(context.dataset_info.sql)
    test_pdf = test_df.to_pandas(all_rows=True)

    X_test = test_pdf[feature_names]
    y_test = test_pdf[target_name]

    print("Scoring")
    y_pred = model.predict(X_test)

    y_pred_tdf = pd.DataFrame(y_pred, columns=[target_name])
    y_pred_tdf["PatientId"] = test_pdf["PatientId"].values

    predictions_table = f"preds_{str(uuid.uuid4()).replace('-','')}"
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    eval_df = DataFrame.from_query(f"""
    SELECT 
        Z.{context.dataset_info.entity_key}, Z.{context.dataset_info.target_names[0]} as Observed, Y.{context.dataset_info.target_names[0]} as Predicted
        FROM ({context.dataset_info.sql}) Z 
        LEFT JOIN (SELECT * FROM {predictions_table}) Y ON Z.{context.dataset_info.entity_key} = Y.{context.dataset_info.entity_key}
    """)

    configure.val_install_location = os.environ.get("AOA_VAL_INSTALL_DB", os.environ.get("VMO_VAL_INSTALL_DB", "TRNG_XSP"))
    statistics = valib.Frequency(data=eval_df, columns='Observed')
    eval_stats = ClassificationEvaluator(data=eval_df, observation_column='Observed', prediction_column='Predicted', num_labels=int(statistics.result.count(True).to_pandas().count_xval))
    eval_data = eval_stats.output_data.to_pandas()

    evaluation = {
        'Accuracy': '{:.2f}'.format(float(eval_data.loc[eval_data['Metric'] == 'Accuracy']['MetricValue'])),
        'Recall': '{:.2f}'.format(float(eval_data.loc[eval_data['Metric'] == 'Macro-Recall']['MetricValue'])),
        'Precision': '{:.2f}'.format(float(eval_data.loc[eval_data['Metric'] == 'Macro-Precision']['MetricValue'])),
        'f1-score': '{:.2f}'.format(float(eval_data.loc[eval_data['Metric'] == 'Macro-F1']['MetricValue']))
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    metrics.plot_confusion_matrix(model, X_test, y_test)
    save_plot('Confusion Matrix', context=context)

    # metrics.plot_roc_curve(model, X_test, y_test)
    # save_plot('ROC Curve', context=context)

    # xgboost has its own feature importance plot support but let's use shap as explainability example
    import shap

    shap_explainer = shap.TreeExplainer(model['xgb'])
    shap_values = shap_explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False, plot_size=(12, 8), plot_type='bar')
    save_plot('SHAP Feature Importance', context=context)

    feature_importance = pd.DataFrame(list(zip(feature_names, np.abs(shap_values).mean(0))),
                                      columns=['col_name', 'feature_importance_vals'])
    feature_importance = feature_importance.set_index("col_name").T.to_dict(orient='records')[0]

    record_evaluation_stats(features_df=test_df,
                            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
                            importance=feature_importance,
                            context=context)
