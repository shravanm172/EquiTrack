from __future__ import annotations

import pandas as pd


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price levels to daily simple returns.

    r_t = (P_t / P_{t-1}) - 1
    """
    returns = prices.pct_change()
    returns = returns.dropna(how="any")
    return returns


def portfolio_returns(asset_returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Compute portfolio daily returns as a weighted sum of asset returns.

    portfolio_return[t] = sum_i (w_i * r_i[t])

    - We normalize weights to sum to 1.0 
    - Missing return values are treated as 0.0 for that day for that asset.
    """
    if asset_returns.empty:
        return pd.Series(dtype="float64", name="portfolio_return")

    # Normalize tickers and build a Series of weights
    w = pd.Series({k.upper(): float(v) for k, v in weights.items()}, dtype="float64")

    # Keep only tickers we actually have return columns for
    cols = [c for c in asset_returns.columns if c.upper() in set(w.index)]
    if not cols:
        raise ValueError("None of the portfolio tickers exist in the market data.")

    w = w.reindex([c.upper() for c in cols])

    # Normalize to sum to 1
    total = float(w.sum())
    if total == 0:
        raise ValueError("Weights sum to 0.")
    w = w / total

    # Apply weights and sum across columns
    r = asset_returns[cols].copy().fillna(0.0)
    # Align weights to the DataFrame columns
    w_aligned = pd.Series([w[c.upper()] for c in cols], index=cols, dtype="float64")

    port_r = (r * w_aligned).sum(axis=1)
    port_r.name = "portfolio_return"
    return port_r