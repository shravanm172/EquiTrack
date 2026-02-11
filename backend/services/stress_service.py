from __future__ import annotations

from typing import Any
import pandas as pd

from providers.market_data import fetch_price_history
from services.analysis_service import _analyze_from_prices, shares_to_weights_from_prices
from services.store_singleton import analysis_store

from engines.scenario_engine import (
    apply_price_shock,
    apply_shock_with_linear_rebound,
    apply_regime_shift,
)


def analyze_with_shock(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Runs baseline analysis and a shocked-scenario analysis, then compares them.

    Supports holdings as either:
      - weights mode: {ticker, weight}
      - shares mode:  {ticker, shares} -> converted to weights using first trading day in date range

    Returns:
      - baseline: equity_curve + metrics
      - scenario: equity_curve + metrics
      - delta: metric differences (scenario - baseline)
    """
    # --- Parse portfolio inputs ---
    portfolio = payload.get("portfolio", {}) or {}
    holdings = portfolio.get("holdings", []) or []

    # Check if user explicitly provided starting cash
    starting_cash_raw = portfolio.get("starting_cash", None)
    starting_cash = float(starting_cash_raw) if starting_cash_raw is not None else None

    # --- Parse date range ---
    date_range = payload.get("date_range", {}) or {}
    start = str(date_range.get("start", "")).strip()
    end = str(date_range.get("end", "")).strip()
    if not start or not end:
        raise ValueError("date_range.start and date_range.end are required.")

    # Decide portfolio mode ONCE (same rule as analyze_portfolio)
    any_shares = any(("shares" in h and h.get("shares") is not None) for h in holdings)
    mode = "shares" if any_shares else "weights"

    shares: dict[str, float] = {}
    weights: dict[str, float] = {}

    # --- Parse holdings + fetch prices once ---
    if mode == "shares":
        for h in holdings:
            ticker = str(h.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            if "shares" not in h or h.get("shares") is None:
                raise ValueError("In shares mode, every holding must include 'shares'.")
            shares[ticker] = float(h.get("shares", 0.0))

        if not shares:
            raise ValueError("Portfolio holdings are required (ticker + shares).")

        ph = fetch_price_history(shares.keys(), start=start, end=end)
        prices = ph.prices

        # Convert shares -> weights using first trading day in window
        weights, holdings_breakdown = shares_to_weights_from_prices(prices, shares)

        # Default starting_cash to market value if omitted
        if starting_cash is None:
            starting_cash = float(holdings_breakdown["total_value"])

    else:
        for h in holdings:
            ticker = str(h.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            weights[ticker] = float(h.get("weight", 0.0))

        if not weights:
            raise ValueError("Portfolio holdings are required (ticker + weight).")

        if starting_cash is None:
            starting_cash = 100_000.0

        ph = fetch_price_history(weights.keys(), start=start, end=end)
        prices = ph.prices

        holdings_breakdown = None

    # --- Parse shock ---
    shock = payload.get("shock", {}) or {}
    shock_date_requested = str(
        shock.get("date")
        or shock.get("shock_date")
        or shock.get("shockDate")
        or ""
    ).strip()
    shock_pct = float(shock.get("pct", 0.0))

    if not shock_date_requested:
        raise ValueError("shock.date is required.")

    shock_date, shock_note = _snap_to_trading_day(prices, shock_date_requested)

    shock_type = str(shock.get("type", "permanent")).strip().lower()
    rebound_days = int(shock.get("rebound_days", 10))

    # --- Baseline analysis ---
    baseline = _analyze_from_prices(prices, weights, float(starting_cash), return_artifacts=True)
    base_art = baseline.pop("_artifacts")

    # --- Scenario analysis (apply shock then re-run analysis) ---
    if shock_type == "permanent":
        shocked_prices = apply_price_shock(
            prices,
            shock_date=shock_date,
            shock_pct=shock_pct,
        )
    elif shock_type == "linear_rebound":
        shocked_prices = apply_shock_with_linear_rebound(
            prices,
            shock_date=shock_date,
            shock_pct=shock_pct,
            rebound_days=rebound_days,
        )
    elif shock_type == "regime_shift":
        vol_mult = float(shock.get("vol_mult", 1.5))
        drift_shift = float(shock.get("drift_shift", -0.0005))
        shocked_prices = apply_regime_shift(
            prices,
            shock_date=shock_date,
            vol_mult=vol_mult,
            drift_shift=drift_shift,
        )
    else:
        raise ValueError(
            f"Unknown shock.type '{shock_type}'. Use 'permanent', 'linear_rebound', or 'regime_shift'."
        )

    scenario = _analyze_from_prices(shocked_prices, weights, float(starting_cash), return_artifacts=True)
    scen_art = scenario.pop("_artifacts")

    # --- Metric deltas (scenario - baseline) ---
    delta_metrics = {
        k: float(scenario["metrics"][k]) - float(baseline["metrics"][k])
        for k in baseline["metrics"].keys()
    }



    resp = {
        "inputs": {
            "mode": mode,
            "starting_cash": float(starting_cash),
            "weights": weights,
            "date_range": {"start": start, "end": end},
            "shock": {
                "date_requested": shock_date_requested,
                "date_applied": shock_date,
                "pct": shock_pct,
                "type": shock_type,
                "note": shock_note,
                "rebound_days": rebound_days if shock_type == "linear_rebound" else None,
                "vol_mult": shock.get("vol_mult") if shock_type == "regime_shift" else None,
                "drift_shift": shock.get("drift_shift") if shock_type == "regime_shift" else None,
            },
        },
        "baseline": baseline,
        "scenario": scenario,
        "delta": {"metrics": delta_metrics},
    }

    # Cache portfolio returns for further analysis
    analysis_id = analysis_store.put({
        "kind": "analyze_shock",
        "inputs": resp["inputs"],
        "baseline_returns": base_art["portfolio_returns"],   # pd.Series
        "scenario_returns": scen_art["portfolio_returns"],   # pd.Series
        "baseline_last_equity_date": base_art["equity_series"].index[-1],
        "baseline_last_equity_value": float(base_art["equity_series"].iloc[-1]),
        "scenario_last_equity_date": scen_art["equity_series"].index[-1],
        "scenario_last_equity_value": float(scen_art["equity_series"].iloc[-1]),
    })
    resp["analysis_id"] = analysis_id

    if mode == "shares" and holdings_breakdown is not None:
        resp["holdings_breakdown"] = holdings_breakdown

    return resp


def _snap_to_trading_day(prices: pd.DataFrame, date_str: str) -> tuple[str, str | None]:
    """
    Helper function: Snap `date_str` (YYYY-MM-DD) to the next trading day in prices.index (>= date_str).

    Returns (applied_date_str, note_or_none).

    Raises ValueError if date_str is invalid or beyond the available window.
    """
    if prices is None or prices.empty:
        raise ValueError("No price data available to align shock date.")

    ts = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(ts):
        raise ValueError("shock.date must be YYYY-MM-DD.")

    idx = prices.index  # DatetimeIndex
    pos = idx.searchsorted(ts, side="left")

    if pos >= len(idx):
        raise ValueError("shock.date is after the last available trading day in the selected window.")

    applied = idx[pos].strftime("%Y-%m-%d")
    note = None
    if applied != date_str:
        note = f"Market closed on {date_str}; applied shock on next trading day {applied}."
    return applied, note