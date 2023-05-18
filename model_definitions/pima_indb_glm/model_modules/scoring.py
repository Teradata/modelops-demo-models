from teradataml import (
    copy_to_sql,
    DataFrame,
    TDGLMPredict,
    ScaleTransform
)
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
import pandas as pd


def score(context: ModelContext, **kwargs):

    aoa_create_context()

    model = DataFrame(f"model_${context.model_version}")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)

    # Scaling the scoring set
    print ("Loading scaler...")
    scaler = DataFrame(f"scaler_${context.model_version}")

    scaled_features = ScaleTransform(
        data=features_tdf,
        object=scaler,
        accumulate = entity_key
    )
    
    print("Scoring")
    predictions = TDGLMPredict(
        object=model,
        newdata=scaled_features.result,
        id_column=entity_key
    )

    predictions_pdf = predictions.result.to_pandas(all_rows=True).rename(columns={"prediction": target_name}).astype(int)

    print("Finished Scoring")

    # store the predictions
    predictions_pdf = pd.DataFrame(predictions_pdf, columns=[target_name])
    predictions_pdf[entity_key] = features_pdf.index.values
    # add job_id column so we know which execution this is from if appended to predictions table
    predictions_pdf["job_id"] = context.job_id

    # teradataml doesn't match column names on append.. and so to match / use same table schema as for byom predict
    # example (see README.md), we must add empty json_report column and change column order manually (v17.0.0.4)
    # CREATE MULTISET TABLE pima_patient_predictions
    # (
    #     job_id VARCHAR(255), -- comes from airflow on job execution
    #     PatientId BIGINT,    -- entity key as it is in the source data
    #     HasDiabetes BIGINT,   -- if model automatically extracts target
    #     json_report CLOB(1048544000) CHARACTER SET UNICODE  -- output of
    # )
    # PRIMARY INDEX ( job_id );
    predictions_pdf["json_report"] = ""
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]

    copy_to_sql(
        df=predictions_pdf,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
        if_exists="append"
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
