from __future__ import annotations

import pandas as pd



def _validate_returns(returns: pd.DataFrame):
    """
    Helper: validates returns DataFrame
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns DataFrame must not be empty.")
    
def ewma_abs_returns(returns:pd.DataFrame, lam:float=0.94)->pd.DataFrame:
    """
    Calculate exponentially weighted moving averages of absolute returns from returns DataFrame
    """
    #Validation
    _validate_returns(returns)
    if not (0.0 < lam < 1.0):
        raise ValueError("lambda must be in (0, 1).")
    
    alpha = 1 - lam
    abs_returns = returns.abs().ewm(alpha=alpha, adjust=False).mean()
    return abs_returns

def rolling_abs_returns(returns:pd.DataFrame, window:int=20)->pd.DataFrame:
    """
    Calculate rolling averages of absolute returns from returns DataFrame
    """
    #Validation
    _validate_returns(returns)
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    
    abs_returns = returns.abs().rolling(window=window).mean()
    return abs_returns

