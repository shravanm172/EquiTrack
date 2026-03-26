from __future__ import annotations

import pandas as pd
import numpy as np


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price levels to daily simple returns.

    r_t = (P_t / P_{t-1}) - 1
    """
    returns = prices.pct_change()
    returns = returns.dropna(how="all")
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

def ewma_correlation_matrix(
    asset_returns: pd.DataFrame,
    lam: float = 0.94,
) -> pd.DataFrame:
    """
    Compute EWMA correlation matrix from daily asset returns.

    IMPORTANT:
    - asset_returns should be daily returns
    - lam must be in (0, 1)
    """
    if asset_returns.empty:
        raise ValueError("asset_returns is empty.")

    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0, 1).")

    returns = asset_returns.dropna(how="any").copy()
    if returns.empty:
        raise ValueError("No overlapping return rows available after dropping NaNs.")

    # Center returns
    centered = returns - returns.mean()

    # Seed with sample covariance
    cov = centered.cov().to_numpy(dtype="float64")

    for _, row in centered.iloc[1:].iterrows():
        r = row.to_numpy(dtype="float64").reshape(-1, 1)
        cov = lam * cov + (1.0 - lam) * (r @ r.T)

    std = np.sqrt(np.diag(cov))
    if np.any(std <= 0):
        raise ValueError("Non-positive variance encountered while computing EWMA correlation.")

    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)

    return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)

def covariance_matrix(
    asset_returns: pd.DataFrame,
    predicted_vols: dict[str, float],
    weights: dict[str, float],
    *,
    corr_method: str="historical",
    lam: float=0.94,
) -> pd.DataFrame:
    """
    Build a hybrid forward looking covariance matrix for the portfolio using predicted future asset volatilities and historical/EWMA correlations

    IMPORTANT: predicted_vols param must be daily (NOT annualized)
    """
    #Validation
    if asset_returns.empty:
        raise ValueError("asset_returns is empty.")

    if not weights:
        raise ValueError("weights is empty.")

    # Normalize ticker casing
    portfolio_tickers = {t.upper() for t in weights.keys()}

    # Keep only relevant columns
    cols = [c for c in asset_returns.columns if c.upper() in portfolio_tickers]

    if not cols:
        raise ValueError("None of the portfolio tickers exist in asset_returns.")

    returns = asset_returns[cols].copy()

    # Compute correlation matrix
    corr_method = corr_method.strip().lower()
    if corr_method == "historical":
        corr_matrix = returns.corr()
    elif corr_method == "ewma":
        corr_matrix = ewma_correlation_matrix(returns, lam=lam)
    else:
        raise ValueError("corr_method must be 'historical' or 'ewma'.")

    # #Compute correlation matrix
    # corr_matrix = returns.corr()

    #Align predicted asset-volatilities with columns
    vol_vector = []

    for col in cols:
        ticker = col.upper()
        if ticker not in predicted_vols:
            raise ValueError(f"Missing predicted volatility for {ticker}")
        vol_vector.append(predicted_vols[ticker])

    vol_vector = np.array(vol_vector)

    #Compute covariance matrix
    vol_matrix = np.outer(vol_vector, vol_vector)
    cov_matrix = corr_matrix.values * vol_matrix
    cov_df = pd.DataFrame(
        cov_matrix,
        index=cols,
        columns=cols,
    )

    return cov_df


def portfolio_volatility(
    cov_matrix: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """
    Computes portfolio volatility from covariance matrix and portfolio weights
    portfolio_variance = w^T Σ w
    portofolio_voltility = sqrt(portfolio_variance)
    """
    if cov_matrix.empty:
        raise ValueError("cov_matrix is empty.")

    if not weights:
        raise ValueError("weights is empty.")
    # Normalize ticker casing in the weight dict
    w = pd.Series({k.upper(): float(v) for k, v in weights.items()}, dtype="float64")
    # Align weights to covariance matrix columns
    cols = list(cov_matrix.columns)
    cols_upper = [c.upper() for c in cols]
    if not any(t in set(cols_upper) for t in w.index):
        raise ValueError("None of the portfolio tickers exist in the covariance matrix.")
    w = w.reindex(cols_upper).fillna(0.0)

    total = float(w.sum())
    if total == 0:
        raise ValueError("Weights sum to 0.")
    w = w / total

    w_vec = w.to_numpy()
    sigma = cov_matrix.to_numpy()

    variance = float(w_vec.T @ sigma @ w_vec)
    volatility = variance ** 0.5
    return volatility


