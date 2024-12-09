
from teradataml import (
    DataFrame,
    ScaleFit,
    ScaleTransform,
)
from teradataml import td_sklearn as osml

from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

from collections import Counter
import shap
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def train(context: ModelContext, **kwargs):
    aoa_create_context()
    
    feature_names = context.dataset_info.feature_names
    entity_key = context.dataset_info.entity_key

    train_df = DataFrame.from_query(context.dataset_info.sql)
    X_train = train_df.drop(['id', 'species'], axis=1)
    
    print("Starting training using teradata osml...")

    isolation_forest_model = osml.IsolationForest(
        n_estimators=context.hyperparams["n_estimators"],
        contamination=context.hyperparams["contamination"],
        random_state=context.hyperparams["random_state"]
    )
    isolation_forest_model.fit(X_train)
    isolation_forest_model.deploy(model_name="Isolation_Forest", replace_if_exists=True)
    print("Complete osml training...")    
    
    print("All done!")
