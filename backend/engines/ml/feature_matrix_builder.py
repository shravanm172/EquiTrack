from __future__ import annotations

import pandas as pd

def _stack_feature(feature_df: pd.DataFrame, feature_name: str) -> pd.Series:
    """
    Helper function: Takes feature DataFrame and converts it to a multi-index Series
    Returns Series with index=(date,ticker)
    """
    stacked = feature_df.stack()
    stacked.name = feature_name
    return stacked

def build_feature_matrix(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Takes a dict with feature names and dataframes and converts to a DataFrame with index=(date,ticker) and column=feature_name
    """
    stacked_features=[]

    for feature_name, feature_df in features.items():
        stacked_features.append(_stack_feature(feature_df, feature_name))
    
    X = pd.concat(stacked_features, axis=1).sort_index()
    return X