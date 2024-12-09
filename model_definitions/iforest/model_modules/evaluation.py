from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from teradataml import td_sklearn as osml
from teradataml import(
    DataFrame, 
    copy_to_sql, 
    get_context, 
    get_connection, 
    ScaleTransform, 
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
import shap
import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import IPython.display
        
def evaluate(context: ModelContext, **kwargs):
    
    aoa_create_context()

    
    feature_names = context.dataset_info.feature_names
    entity_key = context.dataset_info.entity_key

    # Load the test data from Teradata
    test_df = DataFrame.from_query(context.dataset_info.sql)
    X_test = test_df.drop(['id', 'species'], axis=1)
    
    print("Evaluating osml...")
    
    isolation_forest_model = osml.load(model_name="Isolation_Forest")
    predict_df = isolation_forest_model.decision_function(X_test)

    explainer_shap = shap.TreeExplainer(isolation_forest_model.modelObj)
    shap_values = explainer_shap.shap_values(X_test.to_pandas())
    
    # shap.summary_plot(shap_values, X_test.to_pandas(), show=False)
    shap.summary_plot(shap_values, X_test.to_pandas())
    save_plot('SHAP Feature Importance', context=context)
    
#     explainer_ebm = shap.Explainer(isolation_forest_model.modelObj.predict, X_test.to_pandas())
#     shap_values_ebm = explainer_ebm(X_test.to_pandas())
    
#     shap.plots.beeswarm(shap_values_ebm, show=False)
#     save_plot('SHAP Beeswarm Plot', context=context)

    print("All done!")
