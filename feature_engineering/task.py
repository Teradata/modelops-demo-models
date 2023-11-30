from teradataml import DataFrame
from aoa import (aoa_create_context, ModelContext)

def run_task(context: ModelContext, **kwargs):
    aoa_create_context()
    df = DataFrame.from_query("sel age, count(*) as patient_cnt from pima_patient_features group by 1")
    print(df)
    with open(f"{context.artifact_output_path}/age_report.txt", "w") as f:
        print(df, file=f)