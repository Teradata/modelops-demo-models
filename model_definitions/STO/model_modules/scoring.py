from teradataml import DataFrame
from teradatasqlalchemy.types import INTEGER
from collections import OrderedDict
from .util import get_df_with_model
from aoa.util import (
    check_sto_version,
    aoa_create_context,
    ModelContext,
    execute_sql
)

import base64
import dill


def score(context: ModelContext, **kwargs):

    aoa_create_context()

    model_version = context.model_version
    model_table = "vmo_sto_models"

    check_sto_version()

    if model_version == "cli":
        try:
            execute_sql(
                f"DELETE FROM {context.dataset_info.predictions_table} WHERE job_id='cli'")
        except:
            print("Something went wrong trying to cleanup cli model version, maybe it's nothing")

    df = DataFrame.from_query(context.dataset_info.sql)

    def score_partition(partition, features):

        rows = partition.read()

        if rows is None or len(rows) == 0:
            return None

        # the model artefact is available on the 1st row only (see how we joined in the dataframe query)
        model_artefact = rows.loc[rows['n_row'] == 1, 'model'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        out_df = rows[["PatientId"]]
        out_df["prediction"] = model.predict(rows[features])

        return out_df

    number_of_amps = 2
    pdf = df.assign(partition_id=df.PatientId % number_of_amps)
    partitioned_dataset_table = "partitioned_dataset"
    pdf.to_sql(partitioned_dataset_table, if_exists='replace', temporary=True)

    df_with_model = get_df_with_model(
        partitioned_dataset_table, model_table, model_version)

    features = ["NumTimesPrg", "Age", "PlGlcConc", "BloodP",
                "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc"]

    scored_df = df_with_model.map_partition(lambda partition: score_partition(partition, features),
                                            data_partition_column="partition_id",
                                            returns=OrderedDict(
        [('PatientId', INTEGER()),
         ('HasDiabetes', INTEGER())]))

    scored_df = scored_df.assign(job_id=context.job_id, json_report="").select(
        ["job_id", "PatientId", "HasDiabetes", "json_report"])
    scored_df.to_sql(context.dataset_info.predictions_table,
                     if_exists="append")

    print("Finished scoring")
