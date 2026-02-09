from __future__ import annotations

from typing import Any

from providers.market_data import fetch_price_history
from services.analysis_service import _analyze_from_prices


from engines.scenario_engine import (
    apply_price_shock,
    apply_shock_with_linear_rebound,
    apply_regime_shift,
)


def analyze_with_shock(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Runs baseline analysis and a shocked-scenario analysis, then compares them.

    Returns:
      - baseline: equity_curve + metrics
      - scenario: equity_curve + metrics
      - delta: metric differences (scenario - baseline)
    """
    # --- Parse portfolio inputs ---
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

    # --- Parse date range ---
    date_range = payload.get("date_range", {}) or {}
    start = str(date_range.get("start", "")).strip()
    end = str(date_range.get("end", "")).strip()
    if not start or not end:
        raise ValueError("date_range.start and date_range.end are required.")

    # --- Parse shock ---
    shock = payload.get("shock", {}) or {}
    shock_date = str(shock.get("date", "")).strip()
    shock_pct = float(shock.get("pct", 0.0))

    if not shock_date:
        raise ValueError("shock.date is required.")
    # shock_pct can be 0.0 (then scenario == baseline)

    # --- Fetch baseline prices once ---
    ph = fetch_price_history(weights.keys(), start=start, end=end)
    prices = ph.prices

    # --- Baseline analysis ---
    baseline = _analyze_from_prices(prices, weights, starting_cash)

    # --- Scenario analysis (apply shock then re-run analysis) ---
    shock_type = str(shock.get("type", "permanent")).strip().lower()
    rebound_days = int(shock.get("rebound_days", 10))

    if shock_type == "permanent":
        shocked_prices = apply_price_shock(prices, shock_date=shock_date, shock_pct=shock_pct)
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
        raise ValueError(f"Unknown shock.type '{shock_type}'. Use 'permanent' or 'linear_rebound'.")
    scenario = _analyze_from_prices(shocked_prices, weights, starting_cash)

    # --- Metric deltas (scenario - baseline) ---
    delta_metrics = {
        k: float(scenario["metrics"][k]) - float(baseline["metrics"][k])
        for k in baseline["metrics"].keys()
    }

    return {
        "inputs": {
            "starting_cash": starting_cash,
            "weights": weights,
            "date_range": {"start": start, "end": end},
            "shock": {"date": shock_date, "pct": shock_pct},
        },
        "baseline": baseline,
        "scenario": scenario,
        "delta": {"metrics": delta_metrics},
    }
