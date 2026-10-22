# Models

This repository contains the example Diabetes Prediction model code for [AI Studio ModelOps](https://docs.teradata.com/r/Teradata-AI-Studio-Model-Management/ModelOps/Using-ModelOps). We provide a number of demo projects and associated repositories. The goal of these examples is to provide a simple reference implementation for users and not to provide a detailed data science example for each use case. 

Please refer to the [Official Documentation](https://docs.teradata.com/r/Teradata-AI-Studio-Model-Management/ModelOps) for more information.

## Available Models

The example use case we deal with in this demo project is diabetes prediction. More specifically, we use the [PIMA diabetes](http://nrvis.com/data/mldata/pima-indians-diabetes.csv) dataset and build a classification model to predict the likelihood that a given patient has diabetes based on certain medical features.

####  Git

`Git` models are those models where we manage all the code for training evaluation and scoring. We provide Python examples, including notebooks for the Python Diabetes Prediction model.

- [In-database Model Prediction](model_definitions/pima_python_indb_xgboost)
- [Python Diabetes Prediction](model_definitions/python-diabetes)

#### BYOM

`BYOM` is related to [Teradata BYOM](https://docs.teradata.com/r/Enterprise_IntelliFlex_Lake_VMware/Teradata-VantageTM-Bring-Your-Own-Model-User-Guide/Welcome-to-Bring-Your-Own-Model) where we use an open model format such as `PMML` or `ONNX`. In this example, we provide a notebook to produce the `PMML` model artefact along with examples for how to query it dynamically in Teradata. 

- [Diabetes Prediction](byom/pima)

---

_Copyright 2026 Teradata. All Rights Reserved._
