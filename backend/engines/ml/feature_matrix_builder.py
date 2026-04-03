from __future__ import annotations

import pandas as pd

def build_feature_matrix(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build portfolio-level feature matrix with date index and feature-name columns.
    """
    feature_frames = []

    for feature_name, feature_df in features.items():
        df = feature_df.copy()
        df.columns = [feature_name]
        feature_frames.append(df)

    X = pd.concat(feature_frames, axis=1).sort_index()
    return X