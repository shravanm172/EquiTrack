from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_YEAR = 252

eps = 1e-8

def _validate_returns(returns: pd.DataFrame):
    """
    Helper: validates returns DataFrame
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns DataFrame must not be empty.")

def rolling_volatility(returns:pd.DataFrame, window:int=20, annualize:bool=True,)->pd.DataFrame:
    """
        Compute rolling realized volatility from a returns DataFrame
    """
    
     # Validation
    _validate_returns(returns)
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")

    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * (TRADING_DAYS_PER_YEAR ** 0.5)

    return vol

def ewm_volatility(returns:pd.DataFrame, lam:float=0.94 ,annualize:bool=True)->pd.DataFrame:
    """
    Compute exponentially weighted moving volatility over time from a returns DataFrame
    """
    
    # Validation
    _validate_returns(returns)
    if not (0.0 < lam < 1.0):
        raise ValueError("lambda must be in (0, 1).")

    alpha = 1 - lam
    vol = returns.ewm(alpha=alpha, adjust=False).std()

    if annualize:
        vol = vol * (TRADING_DAYS_PER_YEAR ** 0.5)
    
    return vol


def vol_of_vol(vol_df: pd.DataFrame, window: int=20)->pd.DataFrame:
    return vol_df.rolling(window).std() / (vol_df.rolling(window).mean() + eps)

def vol_change(vol_df:pd.DataFrame)->pd.DataFrame:
    return vol_df.pct_change()

def vol_ratio(vol_df:pd.DataFrame, lag:int=5)->pd.DataFrame:
    return vol_df / (vol_df.shift(lag) + eps)

def vol_short_long_ratio(returns:pd.DataFrame)->pd.DataFrame:
    vol_5 = rolling_volatility(returns, 5)
    vol_20 = rolling_volatility(returns, 20)
    return vol_5 / (vol_20 + eps)