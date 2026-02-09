from __future__ import annotations

from typing import Any

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns
from engines.analytics_engine import (
    equity_curve,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)

def _analyze_from_prices(
    prices,
    weights: dict[str, float],
    starting_cash: float,
) -> dict[str, Any]:
    """
    Core analysis pipeline given prices:
      prices -> returns -> portfolio returns -> equity curve -> metrics
    Returns JSON-serializable dict with equity_curve + metrics.
    """
    asset_r = prices_to_returns(prices)
    port_r = portfolio_returns(asset_r, weights)
    curve = equity_curve(port_r, starting_cash)

    metrics = {
        "annualized_return": annualized_return(curve),
        "annualized_volatility": annualized_volatility(port_r),
        "max_drawdown": max_drawdown(curve),
        "sharpe_ratio": sharpe_ratio(port_r),
    }

    curve_json = [
        {"date": idx.strftime("%Y-%m-%d"), "value": round(float(val), 2)}
        for idx, val in curve.items()
    ]

    return {"equity_curve": curve_json, "metrics": metrics}


def analyze_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
    """
    tickers -> prices -> asset returns -> portfolio returns -> equity curve -> risk metrics

    Returns a JSON-serializable dict.
    """

    portfolio = payload.get("portfolio", {}) or {}
    holdings = portfolio.get("holdings", []) or []
    starting_cash = float(portfolio.get("starting_cash", 100000))

    weights: dict[str, float] = {}
    for h in holdings:
        ticker = str(h.get("ticker", "")).strip().upper()
        weight = float(h.get("weight", 0.0))
        if ticker:
            weights[ticker] = weight

    if not weights:
        raise ValueError("Portfolio holdings are required (ticker + weight).")

    date_range = payload.get("date_range", {}) or {}
    start = str(date_range.get("start", "")).strip()
    end = str(date_range.get("end", "")).strip()
    if not start or not end:
        raise ValueError("date_range.start and date_range.end are required.")

    # Execute core analysis pipeline to compute equity curve + metrics baseline
    ph = fetch_price_history(weights.keys(), start=start, end=end)
    prices = ph.prices

    baseline = _analyze_from_prices(prices, weights, starting_cash)

    

    return {
        "inputs": {
            "starting_cash": starting_cash,
            "weights": weights,
            "date_range": {"start": start, "end": end},
        },
        **baseline,
    }
