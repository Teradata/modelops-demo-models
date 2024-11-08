from teradataml import DataFrame

def get_df_with_model(data_table: str,
						 model_artefacts_table: str,
						 model_version: str,
						 partition_id: str = "partition_id"):
	query = f"""
	SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model 
		FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.{partition_id} ORDER BY x.{partition_id}) AS n_row FROM {data_table} x) AS d
		CROSS JOIN {model_artefacts_table} m
		WHERE m.model_version = '{model_version}'
	"""

	return DataFrame.from_query(query)