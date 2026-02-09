from __future__ import annotations
import math

import pandas as pd


def equity_curve(portfolio_returns: pd.Series, starting_cash: float) -> pd.Series:
    """
    Convert portfolio returns into an equity curve (portfolio value over time).

    V_t = starting_cash * Π(1 + r_t)
    """
    if portfolio_returns.empty:
        return pd.Series(dtype="float64", name="equity")

    curve = (1.0 + portfolio_returns).cumprod() * float(starting_cash)
    curve.name = "equity"
    return curve

def annualized_volatility(portfolio_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized volatility = std(daily_returns) * sqrt(252)

    Uses sample std (ddof=1)
    """
    if portfolio_returns.empty:
        return 0.0
    daily_std = float(portfolio_returns.std(ddof=1))
    return daily_std * math.sqrt(periods_per_year)


def max_drawdown(curve: pd.Series) -> float:
    """
    Max Drawdown = minimum of (equity / running_max - 1)
    """
    if curve.empty:
        return 0.0
    running_max = curve.cummax()
    drawdowns = (curve / running_max) - 1.0
    return float(drawdowns.min())


def annualized_return(curve: pd.Series) -> float:
    """
    Annualized return based on start/end value over the time span:
      (end/start)^(1/years) - 1
    """
    if curve.empty:
        return 0.0

    start_val = float(curve.iloc[0])
    end_val = float(curve.iloc[-1])
    if start_val <= 0:
        return 0.0

    days = (curve.index[-1] - curve.index[0]).days
    if days <= 0:
        return 0.0

    years = days / 365.25
    return (end_val / start_val) ** (1.0 / years) - 1.0


def sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe Ratio = (E[R_p - R_f]) / std(R_p)

    - portfolio_returns: daily returns
    - risk_free_rate_annual: annual risk-free rate (e.g. 0.02 = 2%)
    """
    if portfolio_returns.empty:
        return 0.0

    # Convert annual risk-free rate to daily
    rf_daily = risk_free_rate_annual / periods_per_year

    excess_returns = portfolio_returns - rf_daily
    std = excess_returns.std(ddof=1)

    if std == 0 or pd.isna(std):
        return 0.0

    sharpe_daily = excess_returns.mean() / std
    sharpe_annual = sharpe_daily * (periods_per_year ** 0.5)

    return float(sharpe_annual)