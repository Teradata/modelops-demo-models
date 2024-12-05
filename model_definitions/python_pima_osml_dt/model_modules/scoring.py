from teradataml import td_sklearn as osml
from teradataml import (
    copy_to_sql,
    DataFrame,
    ScaleTransform
)
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
import pandas as pd

import json
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def score(context: ModelContext, **kwargs):
    
    aoa_create_context()

    
    # Extract feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key
    
    # Load the test dataset
    test_df = DataFrame.from_query(context.dataset_info.sql)
    copy_to_sql(
        df=test_df,
        schema_name=context.dataset_info.predictions_database,
        table_name='test_df',
        index=False,
        if_exists="replace"
    )
    X_test = test_df.drop(['PatientId'], axis = 1)
    # y_test = test_df.select(["anomaly_int"])
    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)
    test_df.set_index("PatientId")

    print("Scoring using osml...")
    DT_classifier = osml.load(model_name="DT_classifier")
    predict_df =DT_classifier.predict(X_test)
    # Convert predictions to pandas DataFrame and process
    # predictions_pdf = predict_DT.to_pandas(all_rows=True)
    df_pred = predict_df.to_pandas(all_rows=True)
    
    predictions_pdf = predict_df.to_pandas(all_rows=True).rename(columns={"decisiontreeclassifier_predict_1": target_name})
    print("Finished Scoring")

    # store the predictions
   
    predictions_pdf = pd.DataFrame(predictions_pdf, columns=[target_name])
    # predictions_pdf[entity_key] = features_pdf.index.values
    predictions_pdf[entity_key] = test_df.select(["PatientId"]).get_values()
    # add job_id column so we know which execution this is from if appended to predictions table
    # print(predictions_pdf)
    predictions_pdf["job_id"] = context.job_id
    predictions_pdf["json_report"] = ""
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]
    
    copy_to_sql(
        df=predictions_pdf,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
        if_exists="replace"
    )
        
    print("Saved predictions in Teradata")

    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
            WHERE job_id = '{context.job_id}'
    """)
    
    record_scoring_stats(features_df=features_tdf, predicted_df=predictions_df, context=context)

    print("All done!")
