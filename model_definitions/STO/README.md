# STO Micro Modelling Example

This model is an example to show how to use STOs to train, evaluate and score micro models (individual models per data partition in Teradata).

A sample notebook is located [here](notebooks/STO%20examples.ipynb).

## Datasets

The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](http://nrvis.com/data/mldata/pima-indians-diabetes.csv).

```sql
CREATE TABLE PIMA_PATIENT_FEATURES AS 
    (SELECT 
        patientid,
        numtimesprg, 
        plglcconc, 
        bloodp, 
        skinthick, 
        twohourserins, 
        bmi, 
        dipedfunc, 
        age 
    FROM PIMA 
    ) WITH DATA;
    
    
CREATE TABLE PIMA_PATIENT_DIAGNOSES AS 
    (SELECT 
        patientid,
        hasdiabetes
    FROM PIMA 
    ) WITH DATA;
    

```

The micro models are stored in Teradata in the following schema.

```sql
CREATE MULTISET TABLE vmo_sto_models, FALLBACK ,
     NO BEFORE JOURNAL,
     NO AFTER JOURNAL,
     CHECKSUM = DEFAULT,
     DEFAULT MERGEBLOCKRATIO,
     MAP = TD_MAP1
     (
        partition_id VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
        model_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
        num_rows BIGINT,
        partition_metadata JSON CHARACTER SET UNICODE,
        model_artefact CLOB
     )
UNIQUE PRIMARY INDEX (partition_id, model_version);
```
