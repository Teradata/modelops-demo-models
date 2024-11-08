from teradataml import DataFrame
from teradatasqlalchemy.types import INTEGER, VARCHAR, CLOB
from sklearn import metrics
from collections import OrderedDict
from aoa import ModelContext
from .util import get_df_with_model
from aoa.util import (
    save_metadata,
    check_sto_version,
    save_evaluation_metrics,
    aoa_create_context
)

import numpy as np
import json
import base64
import dill


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    model_version = context.model_version
    model_table = "vmo_sto_models"

    check_sto_version()

    df = DataFrame.from_query(context.dataset_info.sql)

    # perform simple feature engineering example using map_row
    def transform_row(row):
        row['Age'] = row['Age'] + 10
        return row

    df = df.map_row(lambda row: transform_row(row))

    def eval_partition(partition):

        rows = partition.read()

        if rows is None or len(rows) == 0:
            return None

        model_artefact = rows.loc[rows['n_row'] == 1, 'model'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        X_test = rows[["NumTimesPrg", "Age", "PlGlcConc", "BloodP",
                       "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc"]]
        y_test = rows[["HasDiabetes"]]

        y_pred = model.predict(X_test)

        # record whatever partition level information you want like rows, data stats, metrics, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": rows.shape[0],
            "metrics": {
                "MAE": "{:.2f}".format(metrics.mean_absolute_error(y_test, y_pred)),
                "MSE": "{:.2f}".format(metrics.mean_squared_error(y_test, y_pred)),
                "R2": "{:.2f}".format(metrics.r2_score(y_test, y_pred))
            }
        })

        partition_id = rows.partition_id.iloc[0]

        # now return a single row for this partition with the evaluation results
        # (schema/order must match returns argument in map_partition)
        return np.array([[partition_id,
                          rows.shape[0],
                          partition_metadata]])

    number_of_amps = 2
    pdf = df.assign(partition_id=df.PatientId % number_of_amps)
    partitioned_dataset_table = "partitioned_dataset"
    pdf.to_sql(partitioned_dataset_table, if_exists='replace', temporary=True)

    df_with_model = get_df_with_model(
        partitioned_dataset_table, model_table, model_version)

    eval_df = df_with_model.map_partition(lambda partition: eval_partition(partition),
                                          data_partition_column="partition_id",
                                          returns=OrderedDict(
        [('partition_id', VARCHAR(255)),
         ('num_rows', INTEGER()),
         ('partition_metadata', CLOB())]))

    # persist to temporary table for computing global metrics
    eval_df.to_sql("sto_eval_results", if_exists="replace", temporary=True)
    eval_df = DataFrame("sto_eval_results")

    save_metadata(eval_df)
    save_evaluation_metrics(eval_df, ["MAE", "MSE", "R2"])

    print("Finished evaluation")
