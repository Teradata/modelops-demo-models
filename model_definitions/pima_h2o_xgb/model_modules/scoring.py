from teradataml import copy_to_sql, get_context, DataFrame, H2OPredict
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    store_byom_tmp,
    ModelContext
)


def score(context: ModelContext, **kwargs):

    aoa_create_context()

    with open(f"{context.artifact_input_path}/model.h2o", "rb") as f:
        model_bytes = f.read()

    model = store_byom_tmp(get_context(), "byom_models_tmp", context.model_version, model_bytes)

    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    byom_target_sql = "CAST(prediction AS INT)"

    print("Scoring")
    h2o = H2OPredict(
        modeldata=model,
        newdata=DataFrame.from_query(context.dataset_info.sql),
        accumulate=context.dataset_info.entity_key)

    print("Finished Scoring")


    # store the predictions
    predictions_df = h2o.result
    
    # add job_id column so we know which execution this is from if appended to predictions table
    predictions_df = predictions_df.assign(job_id=context.job_id)
    cols = {}
    cols[target_name] = predictions_df['prediction']
    predictions_df = predictions_df.assign(**cols)
    predictions_df = predictions_df[["job_id", entity_key, target_name, "json_report"]]

    copy_to_sql(df=predictions_df,
                schema_name=context.dataset_info.predictions_database,
                table_name=context.dataset_info.predictions_table,
                index=False,
                if_exists="append")

    print("Saved predictions in Teradata")
    
    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
            WHERE job_id = '{context.job_id}'
    """)

    record_scoring_stats(features_df=DataFrame.from_query(context.dataset_info.sql),
                         predicted_df=predictions_df,
                         context=context)
