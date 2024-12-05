from teradataml import (
    DataFrame,
    ScaleFit,
    ScaleTransform,
)
from teradataml import td_sklearn as osml

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
import shap
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Compute feature importance based on tree traversal
def compute_feature_importance(model,X_train):
    # from sklearn.inspection import permutation_importance
    feat_dict= {}
    for col, val in sorted(zip(X_train.columns, model.tree_.compute_feature_importances()),key=lambda x:x[1],reverse=True):
        feat_dict[col]=val
    feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
    
    return feat_df
    

def plot_feature_importance(fi, img_filename):
    feat_importances = fi.sort_values(['Importance'],ascending = False).head(10)
    feat_importances.plot(kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()
    
def train(context: ModelContext, **kwargs):
    aoa_create_context()
    
    # Extracting feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the training data from Teradata
    train_df = DataFrame.from_query(context.dataset_info.sql)

    print ("Scaling using InDB Functions...")
    X_train = train_df.drop(['HasDiabetes','PatientId'], axis = 1)
    y_train = train_df.select(["HasDiabetes"])
    # Scale the training data using the ScaleFit and ScaleTransform functions
    scaler = ScaleFit(
        data=train_df,
        target_columns = feature_names,
        scale_method="STD",
        global_scale=False
    )

    scaled_train = ScaleTransform(
        data=train_df,
        object=scaler.output,
        accumulate = [target_name,entity_key]
    )
    
    scaler.output.to_sql(f"scaler_${context.model_version}", if_exists="replace")
    print("Saved scaler")
    
         
    print("Starting training using teradata osml...")

    DT_classifier = osml.DecisionTreeClassifier(random_state=context.hyperparams["random_state"]
                                                ,max_leaf_nodes=context.hyperparams["max_leaf_nodes"]
                                                ,max_features=context.hyperparams["max_features"]
                                                ,max_depth=context.hyperparams["max_depth"])
    DT_classifier.fit(X_train, y_train)
    DT_classifier.deploy(model_name="DT_classifier", replace_if_exists=True)
        
    print("Complete osml training...")
    
    # Calculate feature importance and generate plot
    feature_importance = compute_feature_importance(DT_classifier.modelObj,X_train)
    plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    
    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        feature_importance=feature_importance,
        context=context
    )
    
    print("All done!")
