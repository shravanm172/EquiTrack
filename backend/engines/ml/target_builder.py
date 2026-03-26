from __future__ import annotations

import pandas as pd


def build_target_vector(target_df: pd.DataFrame) -> pd.Series:
    """
    Converts target DataFrame into a MultiIndex Series.
    """
    y = target_df.stack()
    y.name = "target"
    return y