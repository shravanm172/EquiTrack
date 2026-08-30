"""
Two-step regime-selecting stress test:

  1. preview_regime_states  -- fit the GMM on an already-analyzed
     portfolio's cached returns, return each state's stats for the user to
     inspect, and cache what step 2 needs to run simulation.
  2. run_regime_stress_test -- given a previously-previewed regime_id and a
     user-selected state, force every simulated path to start there and
     simulate forward from the cached-fit model
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from services.store_singleton import analysis_store, get_cached_returns_and_starting_cash
from services.serialization import serialize_series, serialize_band_series
from engines.analytics_engine import equity_curve
from engines.regime_engine import calibrate_regime_params, simulate_regime_switching_paths
from engines.simulation_metrics import (
    summarize_terminal_metrics,
    summarize_drawdown_metrics,
    summarize_path_metrics,
    summarize_variance_metrics,
)

N_REGIMES_DEFAULT = 3
N_REGIMES_MAX = 8
TRADING_DAYS_PER_YEAR = 252


def _serialize_regime_stats(regime_stats: pd.DataFrame) -> list[dict[str, Any]]:
    """One entry per state, in state order, no auto-labeling (calm/crisis/etc.),
    just the raw numbers for the user to read and decide from."""
    records = []
    for label, row in regime_stats.sort_index().iterrows():
        records.append({
            "state": int(label),
            "mean_daily_return": float(row["mean"]),
            "daily_volatility": float(row["std"]),
            "annualized_return": float(row["annualized_return"]),
            "annualized_volatility": float(row["annualized_vol"]),
        })
    return records


def _serialize_transition_probs(transition_probs: pd.DataFrame) -> list[list[float]]:
    """Plain row-major nested list, state order preserved (row i = from state i)."""
    ordered = transition_probs.sort_index().sort_index(axis=1)
    return [[float(v) for v in row] for row in ordered.to_numpy()]


def preview_regime_states(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Fit a regime-switching GMM on an already-analyzed portfolio's cached
    returns. Returns each state's stats for display to the user.
    """
    analysis_id = str(payload.get("analysis_id", "")).strip()
    if not analysis_id:
        raise ValueError("analysis_id is required.")

    source = str(payload.get("source", "baseline")).strip().lower()
    if source not in ("baseline", "scenario"):
        raise ValueError("source must be 'baseline' or 'scenario'.")

    n_regimes = int(payload.get("n_regimes", N_REGIMES_DEFAULT))
    if n_regimes < 2 or n_regimes > N_REGIMES_MAX:
        raise ValueError(f"n_regimes must be between 2 and {N_REGIMES_MAX}.")

    port_r, starting_cash, _cached_inputs = get_cached_returns_and_starting_cash(analysis_id, source)
    port_r = port_r.dropna()
    if port_r.empty:
        raise ValueError("Not enough return data to fit regimes (portfolio returns empty).")

    params = calibrate_regime_params(port_r=port_r, n_regimes=n_regimes)

    # Real, dollar-denominated s0, not params.s0
    hist_curve = equity_curve(port_r, starting_cash)
    s0 = float(hist_curve.iloc[-1])
    last_date = hist_curve.index[-1]

    regime_id = analysis_store.put({
        "kind": "regime_preview",
        "source_analysis_id": analysis_id,
        "n_regimes": n_regimes,
        "regime_stats": params.regime_stats,
        "transition_probs": params.transition_probs,
        "s0": s0,
        "last_date": last_date,
        "hist_curve": hist_curve,
    })

    return {
        "regime_id": regime_id,
        "n_regimes": n_regimes,
        "regime_stats": _serialize_regime_stats(params.regime_stats),
        "transition_probs": _serialize_transition_probs(params.transition_probs),
        "current_regime_probs": [float(p) for p in params.current_regime_probs],
    }


def run_regime_stress_test(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Force every simulated path to start in a user-selected regime state
    (from a prior preview_regime_states call) and simulate forward. Uses
    the cached fit so no refitting.
    """
    regime_id = str(payload.get("regime_id", "")).strip()
    if not regime_id:
        raise ValueError("regime_id is required. Call the regime preview endpoint first.")

    cached = analysis_store.get(regime_id)
    if cached is None:
        raise ValueError("regime_id not found or expired. Re-run the regime preview.")
    if cached.get("kind") != "regime_preview":
        raise ValueError("regime_id does not refer to a regime preview.")

    n_regimes = cached["n_regimes"]

    selected_state = payload.get("selected_state")
    if selected_state is None:
        raise ValueError("selected_state is required.")
    selected_state = int(selected_state)
    if not (0 <= selected_state < n_regimes):
        raise ValueError(f"selected_state must be between 0 and {n_regimes - 1}.")

    horizon = int(payload.get("days", 30))
    if horizon <= 0:
        raise ValueError("days must be > 0.")

    n_paths = int(payload.get("simulations", 1000))
    if n_paths <= 0:
        raise ValueError("simulations must be > 0.")

    random_seed = payload.get("random_seed")

    start_regime_probs = np.zeros(n_regimes)
    start_regime_probs[selected_state] = 1.0

    sim = simulate_regime_switching_paths(
        start_regime_probs=start_regime_probs,
        regime_stats=cached["regime_stats"],
        transition_probs=cached["transition_probs"],
        s0=cached["s0"],
        horizon=horizon,
        n_paths=n_paths,
        random_seed=random_seed,
    )

    price_paths = sim["prices"]
    variance_paths = sim["variances"]

    terminal = summarize_terminal_metrics(price_paths)
    drawdown = summarize_drawdown_metrics(price_paths)
    path_metrics = summarize_path_metrics(price_paths)
    variance = summarize_variance_metrics(variance_paths)

    last_date = cached["last_date"]
    future_idx = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    return {
        "inputs": {
            "regime_id": regime_id,
            "selected_state": selected_state,
            "n_regimes": n_regimes,
            "days": horizon,
            "simulations": n_paths,
        },
        "historical_equity_curve": serialize_series(cached["hist_curve"]),
        "forecast_paths": {
            "p10": serialize_band_series(future_idx, path_metrics["p10_path"][1:]),
            "p25": serialize_band_series(future_idx, path_metrics["p25_path"][1:]),
            "p50": serialize_band_series(future_idx, path_metrics["p50_path"][1:]),
            "p75": serialize_band_series(future_idx, path_metrics["p75_path"][1:]),
            "p90": serialize_band_series(future_idx, path_metrics["p90_path"][1:]),
        },
        "terminal": {
            "mean_terminal_value": round(float(terminal["mean_terminal_value"]), 2),
            "median_terminal_value": round(float(terminal["median_terminal_value"]), 2),
            "bear_case": round(float(terminal["bear_case"]), 2),
            "bull_case": round(float(terminal["bull_case"]), 2),
            "probability_of_loss": float(terminal["probability_of_loss"]),
        },
        "drawdown": {
            "median_max_drawdown": float(drawdown["median_max_drawdown"]),
            "prob_drawdown_gt_20": float(drawdown["prob_drawdown_gt_20"]),
        },
        "variance": {
            k: float(v) for k, v in variance.items() if k != "terminal_variances"
        },
    }
