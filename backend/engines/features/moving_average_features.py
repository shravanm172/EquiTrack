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
    
def moving_average_ratio(prices:pd.DataFrame, short_window:int=20, long_window:int=100,)->pd.DataFrame:
    _validate_prices(prices)
    if short_window <= 0 or long_window <= 0:
        raise ValueError("window sizes must be positive.")
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window.")
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()

    ratio = short_ma / long_ma
    return ratio


