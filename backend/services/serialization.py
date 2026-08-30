"""
Generic {date, value} JSON serialization for pandas Series/index+values pairs
used across API responses. Not owned by any one service -- both
forecast_service.py and regime_service.py format their date-indexed output
the same way.
"""
from __future__ import annotations

from typing import Any
import pandas as pd


def serialize_series(series: pd.Series) -> list[dict[str, Any]]:
    return [
        {"date": idx.strftime("%Y-%m-%d"), "value": round(float(val), 2)}
        for idx, val in series.items()
    ]


def serialize_band_series(index: pd.Index, values) -> list[dict[str, Any]]:
    return [
        {"date": idx.strftime("%Y-%m-%d"), "value": round(float(val), 2)}
        for idx, val in zip(index, values)
    ]
