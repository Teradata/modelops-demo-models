from teradataml import DataFrame
from teradatasqlalchemy.types import INTEGER, VARCHAR, CLOB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from aoa import ModelContext
from aoa.util import (
    save_metadata,
    cleanup_cli,
    check_sto_version,
    collect_sto_versions,
    aoa_create_context,
)

import numpy as np
import json
import base64
import dill


def train(context: ModelContext, **kwargs):

    aoa_create_context()

    model_version = context.model_version
    hyperparams = context.hyperparams
    model_artefacts_table = "vmo_sto_models"

    check_sto_version()

    try:
        cleanup_cli(model_version)
    except:
        print("Something went wrong trying to cleanup cli model version, maybe it's nothing")

    # select the training datast via the fold_id
    df = DataFrame.from_query(context.dataset_info.sql)

    # perform simple feature engineering example using map_row
    def transform_row(row):
        row['Age'] = row['Age'] + 10
        return row

    df = df.map_row(lambda row: transform_row(row))

    def train_partition_model(partition, model_version, hyperparams):
        # read all of the rows into memory (we can also process in chunks)
        rows = partition.read()

        # return if partition has no data
        if rows is None or len(rows) == 0:
            return None

        x = rows[["NumTimesPrg", "Age", "PlGlcConc", "BloodP",
                  "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc"]]
        y = rows[["HasDiabetes"]]

        model = Pipeline([('scaler', MinMaxScaler()),
                          ('xgb', XGBClassifier(eta=hyperparams["eta"], max_depth=hyperparams["max_depth"]))])

        model.fit(x, y)

        partition_id = rows.partition_id.iloc[0]

        partition_metadata = json.dumps({
            "num_rows": rows.shape[0],
            "hyper_parameters": hyperparams
        })

        # we have to convert the model to base64 to store in a CLOB column (can't use BLOB with STOs)
        artefact = base64.b64encode(dill.dumps(model))

        # here we return 1 row per partition - basically, the trained model for that partitiom
        return np.array([[partition_id,
                          model_version,
                          rows.shape[0],
                          partition_metadata,
                          artefact]])

    print("Starting training...")

    number_of_amps = 2
    pdf = df.assign(partition_id=df.PatientId % number_of_amps)
    partitioned_dataset_table = "partitioned_dataset"
    pdf.to_sql(partitioned_dataset_table, if_exists='replace', temporary=True)

    train_df = DataFrame('partitioned_dataset')

    model_df = train_df.map_partition(lambda partition: train_partition_model(partition, model_version, hyperparams),
                                      data_partition_column="partition_id",
                                      returns=OrderedDict(
        [('partition_id', VARCHAR(255)),
         ('model_version', VARCHAR(255)),
         ('num_rows', INTEGER()),
         ('partition_metadata', CLOB()),
         ('model_artefact', CLOB())]))

    model_df.to_sql(model_artefacts_table, if_exists="replace")
    model_df = DataFrame(
        query=f"SELECT * FROM {model_artefacts_table} WHERE model_version='{model_version}'")

    save_metadata(model_df)

    print("Finished training")

    with open(f"{context.artifact_output_path}/sto_versions.json", "w+") as f:
        json.dump(collect_sto_versions(), f)
