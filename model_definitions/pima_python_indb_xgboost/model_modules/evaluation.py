from sklearn import metrics
from teradataml import (
    DataFrame,
    copy_to_sql,
    get_context,
    get_connection,
    ScaleTransform,
    XGBoostPredict,
    ConvertTo,
    ClassificationEvaluator,
    ROC
)
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib
import json
import numpy as np
import pandas as pd
import os


def traverse_tree(tree, feature_counter):
    if 'split_' in tree and 'attr_' in tree['split_']:
        feature_counter[tree['split_']['attr_']] += 1
    if 'leftChild_' in tree:
        traverse_tree(tree['leftChild_'], feature_counter)
    if 'rightChild_' in tree:
        traverse_tree(tree['rightChild_'], feature_counter)


def compute_feature_importance(trees_json):
    feature_counter = Counter()
    for tree_json in trees_json:
        tree = json.loads(tree_json)
        traverse_tree(tree, feature_counter)
    total_splits = sum(feature_counter.values())
    feature_importance = {
        feature: count / total_splits for feature, count in feature_counter.items()}
    return feature_importance


def plot_feature_importance(fi, img_filename):
    import pandas as pd
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(
        kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    print(f"Loading model from table model_{context.model_version}")
    model = DataFrame(f"model_{context.model_version}")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    test_df = DataFrame.from_query(context.dataset_info.sql)

    # Scaling the test set
    print(f"Loading scaler from table scaler_{context.model_version}")
    scaler = DataFrame(f"scaler_{context.model_version}")

    scaled_test = ScaleTransform(
        data=test_df,
        object=scaler,
        accumulate=[target_name, entity_key]
    )

    print("Evaluating...")
    predictions = XGBoostPredict(
        object=model,
        newdata=scaled_test.result,
        model_type='Classification',
        accumulate=target_name,
        id_column=entity_key,
        output_prob=True,
        output_responses=['0', '1'],
        object_order_column=['task_index', 'tree_num', 'iter', 'class_num', 'tree_order']
    )

    predicted_data = ConvertTo(
        data=predictions.result,
        target_columns=[target_name, 'Prediction'],
        target_datatype=["INTEGER"]
    )

    ClassificationEvaluator_obj = ClassificationEvaluator(
        data=predicted_data.result,
        observation_column=target_name,
        prediction_column='Prediction',
        num_labels=2
    )

    metrics_pd = ClassificationEvaluator_obj.output_data.to_pandas()

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics_pd.MetricValue[0]),
        'Micro-Precision': '{:.2f}'.format(metrics_pd.MetricValue[1]),
        'Micro-Recall': '{:.2f}'.format(metrics_pd.MetricValue[2]),
        'Micro-F1': '{:.2f}'.format(metrics_pd.MetricValue[3]),
        'Macro-Precision': '{:.2f}'.format(metrics_pd.MetricValue[4]),
        'Macro-Recall': '{:.2f}'.format(metrics_pd.MetricValue[5]),
        'Macro-F1': '{:.2f}'.format(metrics_pd.MetricValue[6]),
        'Weighted-Precision': '{:.2f}'.format(metrics_pd.MetricValue[7]),
        'Weighted-Recall': '{:.2f}'.format(metrics_pd.MetricValue[8]),
        'Weighted-F1': '{:.2f}'.format(metrics_pd.MetricValue[9]),
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    y_test = predicted_data.result.to_pandas()['HasDiabetes']
    y_pred = predicted_data.result.to_pandas()['Prediction']
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    save_plot('Confusion Matrix', context=context)

    metrics.RocCurveDisplay.from_predictions(y_test, y_pred)
    save_plot('ROC Curve', context=context)

    # Calculate feature importance and generate plot
    try:
        model_pdf = model.result.to_pandas()['classification_tree']
        feature_importance = compute_feature_importance(model_pdf)
        feature_importance_df = pd.DataFrame(
            list(feature_importance.items()), columns=['Feature', 'Importance'])
        plot_feature_importance(
            feature_importance, f"{context.artifact_output_path}/feature_importance")
    except:
        feature_importance = {}

    predictions_table = "predictions_tmp"
    copy_to_sql(df=predicted_data.result, table_name=predictions_table,
                index=False, if_exists="replace", temporary=True)

    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(
            features_df=test_df,
            predicted_df=DataFrame.from_query(
                f"SELECT * FROM {predictions_table}"),
            feature_importance=feature_importance,
            context=context
        )

    print("All done!")
