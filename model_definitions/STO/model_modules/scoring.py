from teradataml import DataFrame
from teradatasqlalchemy.types import INTEGER, VARCHAR, CLOB
from collections import OrderedDict
from .util import get_df_with_model
from aoa import record_scoring_stats
from aoa.util import (
	save_metadata,
	cleanup_cli,
	check_sto_version,
	collect_sto_versions,
	save_evaluation_metrics,
	aoa_create_context,
	ModelContext
)

import base64
import dill

def score(context: ModelContext, **kwargs):

	aoa_create_context()
	
	model_version = context.model_version
	model_table = context.model_table

	check_sto_version()

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
	pdf = df.assign(partition_id = df.PatientId % number_of_amps)
	partitioned_dataset_table = "partitioned_dataset"
	pdf.to_sql(partitioned_dataset_table,if_exists ='replace',temporary=True)

	df_with_model = get_df_with_model(partitioned_dataset_table,model_table,model_version)

	features = ["NumTimesPrg", "Age", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc"]

	scored_df = df_with_model.map_partition(lambda partition: score_partition(partition, features), 
										data_partition_column="partition_id",
										returns=OrderedDict(
										[('PatientId', INTEGER()),
										('prediction', INTEGER())]))
	

	scored_df.to_sql(context.dataset_info.predictions_table, if_exists="replace")

	print(scored_df)