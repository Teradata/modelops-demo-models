from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from teradataml import td_sklearn as osml
# from lime.lime_tabular import LimeTabularExplainer
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

        
# Compute feature importance based on tree traversal
def compute_feature_importance(model,X_train):
    feat_dict= {}
    for col, val in sorted(zip(X_train.columns, model.feature_importances_),key=lambda x:x[1],reverse=True):
        feat_dict[col]=val
    feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})
    # print(feat_df)
    return feat_df

def plot_feature_importance(fi, img_filename):
    feat_importances = fi.sort_values(['Importance'],ascending = False).head(10)
    feat_importances.plot(kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


# Define function to plot a confusion matrix from given data
def plot_confusion_matrix(cf, img_filename):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cf, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cf.shape[0]):
        for j in range(cf.shape[1]):
            ax.text(x=j, y=i,s=cf[i, j], va='center', ha='center', size='xx-large')
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix');
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


# Define function to plot ROC curve from ROC output data 
def plot_roc_curve(roc_out, img_filename):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(roc_out['HasDiabetes'], roc_out['decisiontreeclassifier_predict_1'])
    plt.plot(fpr,tpr,label="ROC curve AUC="+str(auc), color='darkorange')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=200)
    plt.clf()
    
    

def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Load the test data from Teradata
    test_df = DataFrame.from_query(context.dataset_info.sql)
    X_test = test_df.drop(['HasDiabetes','PatientId'], axis = 1)
    y_test = test_df.select(["HasDiabetes"])
    # Scaling the test set
    print ("Loading scaler...")
    scaler = DataFrame(f"scaler_${context.model_version}")

    scaled_test = ScaleTransform(
        data=test_df,
        object=scaler,
        accumulate = [target_name,entity_key]
    )
    
    print("Evaluating osml...")
    DT_classifier = osml.load(model_name="DT_classifier")
    predict_df =DT_classifier.predict(X_test,y_test)
    accuracy_dt = DT_classifier.score(X_test, y_test)
    df = X_test.sample(n=1)
    df = df.drop(columns="sampleid")
   
    explainer_shap = shap.TreeExplainer(DT_classifier.modelObj)
    shap_values = explainer_shap.shap_values(X_test.to_pandas())
    
    shap.summary_plot(shap_values, X_test.to_pandas(),show=False, plot_size=(12, 8), plot_type='bar')
    save_plot('SHAP Feature Importance', context=context)
    
    
    explainer_ebm = shap.Explainer(DT_classifier.modelObj.predict, X_test.to_pandas())
    shap_values_ebm = explainer_ebm(X_test.to_pandas())
    
    shap.plots.beeswarm(shap_values_ebm,show=False, plot_size=(12,8))
    save_plot('SHAP Beeswarm Plot', context=context)

    # Evaluate classification metrics using ClassificationEvaluator
    ClassificationEvaluator_obj = ClassificationEvaluator(
        data=predict_df,
        observation_column=target_name,
        prediction_column='decisiontreeclassifier_predict_1',
        num_labels=2
    )

#      # Extract and store evaluation metrics
    metrics_pd = ClassificationEvaluator_obj.output_data.to_pandas()
      
         
    evaluation = {
        'Accuracy': '{:.4f}'.format(metrics_pd.MetricValue[0]),
        'Micro-Precision': '{:.4f}'.format(metrics_pd.MetricValue[1]),
        'Micro-Recall': '{:.4f}'.format(metrics_pd.MetricValue[2]),
        'Micro-F1': '{:.4f}'.format(metrics_pd.MetricValue[3]),
        'Macro-Precision': '{:.4f}'.format(metrics_pd.MetricValue[4]),
        'Macro-Recall': '{:.4f}'.format(metrics_pd.MetricValue[5]),
        'Macro-F1': '{:.4f}'.format(metrics_pd.MetricValue[6]),
        'Weighted-Precision': '{:.4f}'.format(metrics_pd.MetricValue[7]),
        'Weighted-Recall': '{:.4f}'.format(metrics_pd.MetricValue[8]),
        'Weighted-F1': '{:.4f}'.format(metrics_pd.MetricValue[9]),
        # 'Accuracy-osml': '{:.2f}'.format(accuracy_osml.score[0]),
    }

     # Save evaluation metrics to a JSON file
    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)
        
    # Generate and save confusion matrix plot
    cm = confusion_matrix(predict_df.to_pandas()['HasDiabetes'], predict_df.to_pandas()['decisiontreeclassifier_predict_1'])
    plot_confusion_matrix(cm, f"{context.artifact_output_path}/confusion_matrix")
    
    # Generate and save ROC curve plot
    roc_out = ROC(
        data=predict_df,
        probability_column='decisiontreeclassifier_predict_1',
        observation_column=target_name,
        positive_class='1',
        num_thresholds=1000
    )
    
    plot_roc_curve(predict_df.to_pandas(), f"{context.artifact_output_path}/roc_curve")
    
    feature_importance = compute_feature_importance(DT_classifier.modelObj,X_test)
    plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")
    
    
    predictions_table = "predictions_tmp"
    copy_to_sql(df=predict_df, table_name=predictions_table, index=False, if_exists="replace", temporary=True)
    
    
    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(
            features_df=test_df,
            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
            feature_importance=feature_importance,
            context=context
        )

    print("All done!")
