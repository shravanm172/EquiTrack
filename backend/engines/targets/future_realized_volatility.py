from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _validate_returns(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns DataFrame must not be empty.")


def future_realized_volatility(
    returns: pd.DataFrame,
    horizon: int = 10,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute next-horizon realized volatility target from a returns DataFrame.
    """
    _validate_returns(returns)

    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    target = returns.shift(-1).rolling(window=horizon).std()

    if annualize:
        target = target * (TRADING_DAYS_PER_YEAR ** 0.5)

    return target.shift(-(horizon - 1))