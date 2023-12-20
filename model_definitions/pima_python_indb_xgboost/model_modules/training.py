from teradataml import (
    DataFrame,
    XGBoost,
    ScaleFit,
    ScaleTransform,
)

from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from collections import Counter


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
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(
        kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)

    print("Scaling using InDB Functions...")

    # Extract and cast hyperparameters
    scale_method = str(context.hyperparams["scale_method"])
    miss_value = str(context.hyperparams["miss_value"])
    global_scale = str(context.hyperparams["global_scale"]).lower() in ['true', '1']
    multiplier = str(context.hyperparams["multiplier"])
    intercept = str(context.hyperparams["intercept"])
    model_type = str(context.hyperparams["model_type"])
    lambda1 = float(context.hyperparams["Lambda1"])

    scaler = ScaleFit(
        data=train_df,
        target_columns=feature_names,
        scale_method=scale_method,
        miss_value=miss_value,
        global_scale=global_scale,
        multiplier=multiplier,
        intercept=intercept
    )

    scaled_train = ScaleTransform(
        data=train_df,
        object=scaler.output,
        accumulate=[target_name, entity_key]
    )

    scaler.output.to_sql(
        f"scaler_${context.model_version}", if_exists="replace")
    print("Saved scaler")

    print("Starting training...")

    model = XGBoost(
        data=scaled_train.result,
        input_columns=feature_names,
        response_column=target_name,
        model_type=model_type,
        lambda1=lambda1
    )

    model.result.to_sql(f"model_${context.model_version}", if_exists="replace")
    print("Saved trained model")

    # Calculate feature importance and generate plot
    model_pdf = model.result.to_pandas()['classification_tree']
    feature_importance = compute_feature_importance(model_pdf)
    plot_feature_importance(
        feature_importance, f"{context.artifact_output_path}/feature_importance")

    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        feature_importance=feature_importance,
        context=context
    )

    print("All done!")
