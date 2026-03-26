from __future__ import annotations

import pandas as pd



def _validate_prices(prices: pd.DataFrame):
    """
    Helper: validates returns DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty:
        raise ValueError("prices DataFrame must not be empty.")
    
def drawdown(prices: pd.DataFrame)->pd.DataFrame:
    """
    Compute drawdown over time from a prices DataFrame
    """
    _validate_prices(prices)

    running_peaks = prices.cummax()
    drawdowns = (prices - running_peaks)/running_peaks
    return drawdowns