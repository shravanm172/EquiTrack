from __future__ import annotations

import pandas as pd


def build_target_vector(target_df: pd.DataFrame) -> pd.Series:
    """
    Convert portfolio-level target DataFrame into a date-indexed Series.
    """
    y = target_df.iloc[:, 0].copy()
    y.name = "target"
    return y