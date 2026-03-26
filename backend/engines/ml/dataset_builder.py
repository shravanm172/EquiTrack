from __future__ import annotations
from engines.ml.feature_matrix_builder import build_feature_matrix
from engines.ml.target_builder import build_target_vector

import pandas as pd

def assemble_dataset(features:dict[str, pd.DataFrame], target_df:pd.DataFrame)->tuple[pd.DataFrame, pd.Series]:
    """
        Builds final dataset for model (X,Y)
    """
    X = build_feature_matrix(features)
    y = build_target_vector(target_df)

    dataset = pd.concat([X,y], axis=1)

    # Clean dataset
    dataset = dataset.dropna()

    #Split back
    X_clean = dataset.drop(columns=["target"])
    y_clean = dataset["target"]

    return X_clean, y_clean
    