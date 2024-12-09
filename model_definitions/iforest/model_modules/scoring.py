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

import IPython.display

def score(context: ModelContext, **kwargs):
    
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    entity_key = context.dataset_info.entity_key
    
    test_df = DataFrame.from_query(context.dataset_info.sql)
    X_test = test_df.drop(['id', 'species'], axis=1)
    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)
    test_df.set_index("id")

    print("Scoring using osml...")
    isolation_forest_model = osml.load(model_name="Isolation_Forest")
    predict_df =isolation_forest_model.decision_function(X_test)    
    
    print("Finished Scoring")
    
    pred_col = 'isolationforest_decision_function_1'
   
    predictions_pdf = pd.DataFrame(predict_df.to_pandas(), columns=[pred_col])
    predictions_pdf[entity_key] = test_df.select(["id"]).get_values()
    
    # add job_id column so we know which execution this is from if appended to predictions table
    predictions_pdf["job_id"] = context.job_id
    predictions_pdf["json_report"] = ""
    
    predictions_pdf = predictions_pdf[["job_id", entity_key, pred_col, "json_report"]]
    
    copy_to_sql(
        df=predictions_pdf,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
        if_exists="replace"
    )
        
    print("Saved predictions in Teradata")

    print("All done!")
