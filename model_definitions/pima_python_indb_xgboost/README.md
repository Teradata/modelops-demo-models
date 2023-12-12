# Overview

PIMA Diabetes demo model using Teradata in-database XGBoost model

# Datasets 
The dataset required to train or evaluate this model is the PIMA Indians Diabetes dataset available [here](http://nrvis.com/data/mldata/pima-indians-diabetes.csv).

## Model Configuration

The model's configuration, including its hyperparameters, can be found in the [config.json](model_definitions/pima_python_indb_xgboost/config.json) file.

### Hyperparameters

These hyperparameters can be adjusted to tune the model's performance. Please note that incorrect types or values for these hyperparameters may lead to errors or suboptimal model performance.

The model uses the following hyperparameters:

## Data Scaling

- `scale_method`: The method used for scaling the data. This should be a string.
- `miss_value`: The value used to replace missing values in the data. This should be a float.
- `global_scale`: Whether to scale the data globally. This should be a boolean.
- `multiplier`: The multiplier used in scaling the data. This should be a float.
- `intercept`: The intercept used in scaling the data. This should be a float.

## Model Training

- `max_depth`: The maximum depth of the trees for the XGBoost model. This should be an integer.
- `num_boosted_trees`: The number of boosted trees for the XGBoost model. This should be an integer.
- `tree_size`: The size of the trees for the XGBoost model. This should be a float.
- `lambda1`: The lambda parameter for the XGBoost model. This should be a float.

## Training

To train the model, run the [training.py](model_definitions/pima_python_indb_xgboost/model_modules/training.py) script.

The training function takes the following shape:

```python
def train(context: ModelContext, **kwargs):
    aoa_create_context()
    
    # your training code using teradataml indDB function
    model = <InDB Function>(...)
    
    # save your model
    model.result.to_sql(f"model_${context.model_version}", if_exists="replace")  
    
    record_training_stats(...)
```

## Evaluation

To evaluate the model, run the [evaluation.py](model_definitions/pima_python_indb_xgboost/model_modules/evaluation.py) script.

The evaluation function returns the following metrics

- Accuracy
- Micro-Precision
- Micro-Recall
- Micro-F1
- Macro-Precision
- Macro-Recall
- Macro-F1
- Weighted-Precision
- Weighted-Recall
- Weighted-F1

We produce a number of plots for each evaluation also

- Confusion Matrix
- ROC Curve
- Feature Importance

The evaluation function takes the following shape:

```python
def evaluate(context: ModelContext, **kwargs):
    aoa_create_context()

    # read your model from Vantage
    model = DataFrame(f"model_${context.model_version}")
    
    # your evaluation logic
    
    record_evaluation_stats(...)
```

## Scoring

To score the model, run the [scoring.py](model_definitions/pima_python_indb_xgboost/model_modules/scoring.py) script.

The scoring function takes the following shape:

```python
def score(context: ModelContext, **kwargs):
    aoa_create_context()

    # read your model
    model = DataFrame(f"model_${context.model_version}")
    
    # your evaluation logic
    
    record_scoring_stats(...)
```