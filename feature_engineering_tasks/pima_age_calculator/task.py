from teradataml import DataFrame, copy_to_sql
from aoa import (aoa_create_context, ModelContext)
import pandas as pd
import numpy as np

def run_task(context: ModelContext, **kwargs):
    aoa_create_context()
    df = DataFrame.from_query("sel * from pima_patient_features")
    
    # Convert teradataml DataFrame to pandas DataFrame
    df_pd = df.to_pandas()
    
    # Generate random birthdates
    start_date = pd.to_datetime('1950-01-01')
    end_date = pd.to_datetime('2000-01-01')

    df_pd['birthday'] = start_date + (end_date - start_date) * np.random.rand(len(df_pd))
    
    # Calculate age
    df_pd['calculated_age'] = df_pd['birthday'].apply(lambda x: (pd.to_datetime('today') - x).days // 365)
    
    # Remove the original age column
    df_pd = df_pd.drop(columns=['Age'])
    
    # Write pandas DataFrame to a Teradata table
    copy_to_sql(df = df_pd, table_name = 'age_table', if_exists = 'replace')
    
    # Create a teradataml DataFrame from the table
    df = DataFrame('age_table')
    
    print(df)
    with open(f"{context.artifact_output_path}/age_report.txt", "w") as f:
        print(df, file=f)
    
    # Store build properties as a file artifact
    with open(f"{context.artifact_output_path}/build_properties.txt", "w") as f:
        f.write(str(kwargs))
        