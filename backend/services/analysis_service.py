from __future__ import annotations

from typing import Any, Tuple
import pandas as pd

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns
from engines.analytics_engine import (
    equity_curve,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)
from services.store_singleton import analysis_store


def _analyze_from_prices(
    prices: pd.DataFrame,
    weights: dict[str, float],
    starting_cash: float,
    return_artifacts: bool = False,
) -> dict[str, Any]:
    """
    Core analysis pipeline given prices:
      prices -> returns -> portfolio returns -> equity curve -> metrics
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

    out = {"equity_curve": curve_json, "metrics": metrics}

    if return_artifacts:
        out["_artifacts"] = {
            "portfolio_returns": port_r.dropna(),  # pd.Series
            "equity_series": curve,                
        }

    return out


def analyze_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Supports holdings as either:
      - weights mode: {ticker, weight}
      - shares mode:  {ticker, shares} -> converted to weights using first trading day in date range

    Mixed mode not yet supported. If any holding has 'shares', we treat it as shares mode by default.
    """

    portfolio = payload.get("portfolio", {}) or {}
    holdings = portfolio.get("holdings", []) or []

    # Check if user explicitly provided starting cash
    starting_cash_raw = portfolio.get("starting_cash", None)
    starting_cash = float(starting_cash_raw) if starting_cash_raw is not None else None

    date_range = payload.get("date_range", {}) or {}
    start = str(date_range.get("start", "")).strip()
    end = str(date_range.get("end", "")).strip()
    if not start or not end:
        raise ValueError("date_range.start and date_range.end are required.")

    # Decide portfolio mode ONCE
    any_shares = any(("shares" in h and h.get("shares") is not None) for h in holdings)
    mode = "shares" if any_shares else "weights"

    # Parse holdings accordingly
    shares: dict[str, float] = {}
    weights: dict[str, float] = {}

    if mode == "shares":
        # -------------------------
        # SHARES PATH 
        # -------------------------
        for h in holdings:
            ticker = str(h.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            if "shares" not in h or h.get("shares") is None:
                raise ValueError("In shares mode, every holding must include 'shares'.")
            shares[ticker] = float(h.get("shares", 0.0))

        if not shares:
            raise ValueError("Portfolio holdings are required (ticker + shares).")

        # Fetch prices for tickers (needed both to value shares and run analysis)
        ph = fetch_price_history(shares.keys(), start=start, end=end)
        prices = ph.prices

        # Convert shares -> weights using first trading day in `prices`
        weights, holdings_breakdown = shares_to_weights_from_prices(prices, shares)

        # Default starting_cash to portfolio market value if user didn't provide it
        if starting_cash is None:
            starting_cash = float(holdings_breakdown["total_value"])

    else:
        # --------------------------
        # WEIGHTS PATH 
        # --------------------------
        for h in holdings:
            ticker = str(h.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            weights[ticker] = float(h.get("weight", 0.0))

        if not weights:
            raise ValueError("Portfolio holdings are required (ticker + weight).")

        # Default starting_cash if omitted 
        if starting_cash is None:
            starting_cash = 100_000.0

        # Fetch prices for tickers
        ph = fetch_price_history(weights.keys(), start=start, end=end)
        prices = ph.prices

        holdings_breakdown = None  # not applicable in weights mode

    # Run the unchanged analysis pipeline
    baseline = _analyze_from_prices(prices, weights, float(starting_cash), return_artifacts=True)
    art = baseline.pop("_artifacts")

    analysis_id = analysis_store.put({
        "kind": "analyze",
        "inputs": {
            "mode": mode,
            "starting_cash": float(starting_cash),
            "date_range": {"start": start, "end": end},
            "weights": weights,
        },
        "portfolio_returns": art["portfolio_returns"],  # pd.Series
        "last_equity_date": art["equity_series"].index[-1],
        "last_equity_value": float(art["equity_series"].iloc[-1]),
    })

    resp = {
        "analysis_id": analysis_id,
        "inputs": {
            "mode": mode,
            "starting_cash": float(starting_cash),
            "date_range": {"start": start, "end": end},
            "weights": weights,  # always include computed weights for transparency
        },
        **baseline,
    }

    if mode == "shares":
        resp["holdings_breakdown"] = holdings_breakdown

    return resp


def shares_to_weights_from_prices(
    prices: pd.DataFrame,
    shares_by_ticker: dict[str, float],
) -> Tuple[dict[str, float], dict[str, Any]]:
    """
    Convert shares -> weights using prices on the first available trading day
    in the provided `prices` DataFrame.
    """
    if prices is None or prices.empty:
        raise ValueError("No price data available to value share holdings.")

    as_of_dt = prices.index[0]
    as_of = as_of_dt.strftime("%Y-%m-%d")
    row = prices.iloc[0]

    positions = []
    total_value = 0.0

    for t, sh in shares_by_ticker.items():
        ticker = t.strip().upper()
        shares = float(sh)

        if shares <= 0:
            raise ValueError(f"Shares must be > 0 for {ticker}.")

        if ticker not in prices.columns:
            raise ValueError(f"Ticker {ticker} not found in market data columns.")

        px = row[ticker]
        if pd.isna(px) or float(px) <= 0:
            raise ValueError(f"Missing/invalid price for {ticker} on {as_of}.")

        value = shares * float(px)
        total_value += value

        positions.append(
            {
                "ticker": ticker,
                "shares": round(shares, 6),
                "price": round(float(px), 6),
                "value": round(value, 2),
            }
        )

    if total_value <= 0:
        raise ValueError("Total portfolio value is 0; cannot compute weights.")

    weights: dict[str, float] = {}
    for p in positions:
        w = float(p["value"]) / total_value
        p["weight"] = round(w, 6)
        weights[p["ticker"]] = w

    breakdown = {
        "as_of": as_of,
        "total_value": round(total_value, 2),
        "positions": positions,
    }
    return weights, breakdown
