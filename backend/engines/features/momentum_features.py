from __future__ import annotations

import pandas as pd


def _validate_returns(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns DataFrame must not be empty.")


def momentum(
    returns: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling momentum as the mean return over a recent window.
    """
    _validate_returns(returns)

    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")

    return returns.rolling(window=window).mean()


def momentum_swings(
    returns: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Compute momentum swings as the absolute day-to-day change
    in rolling momentum.

    Large values indicate rapidly changing directional behavior.
    """
    mom = momentum(returns, window=window)
    return mom.diff().abs()