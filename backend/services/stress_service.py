"""
Stress testing: simulate a portfolio forward, under a shocked scenario, from
a chosen as-of date (any trading day within the cached analysis window, up
to and including its last date).

Two shock mechanisms, same response shape:
  - deterministic_regime: user-specified drift_shift/vol_mult applied to the
    real historical mu/sigma estimated up to the as-of date, then a plain
    shocked-GBM Monte Carlo forward from there.
  - calibrated_regime: a GMM regime-switching fit (two-step -- preview the
    fitted states, then force every simulated path to start in a
    user-selected state) forward from the as-of date.

Both are pure forward simulations from cached returns -- neither refetches
price data. Both funnel through run_stochastic_forecast (no direct engine
calls here) and _build_stress_response (one shared response shape), so the
only thing that differs between them is how the simulation's params get
prepared.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np
import pandas as pd

from services.store_singleton import analysis_store, get_cached_returns_and_starting_cash
from services.serialization import serialize_series, serialize_band_series
from engines.analytics_engine import equity_curve
from engines.forecast_estimators import estimate_drift, estimate_volatility
from engines.stochastic_engine import run_stochastic_forecast
from engines.regime_engine import calibrate_regime_params, RegimeCalibratedParams

TRADING_DAYS_PER_YEAR = 252
N_REGIMES_DEFAULT = 3
N_REGIMES_MAX = 8
DETERMINISTIC_DRIFT_SHIFT_DEFAULT = -0.0005
DETERMINISTIC_VOL_MULT_DEFAULT = 1.5

def _truncate_returns(port_r: pd.Series, as_of_date_requested: str | None) -> tuple[pd.Series, str, str | None]:
    """
    Snap as_of_date to the next available trading day in port_r's index
    (>= requested), then truncate port_r to that date inclusive,
    so that the simulation logic downstream never sees data past the as_of_date.
    
    Parameters:
        - port_r: The portfolio returns series for the whole user-defined analysis window
        - as_of_date_requested: The requested date from which to start the stress analysis

    Returns:
        - The truncated portfolio returns series
        - The actual date from which the stress analysis begins
        - Message indicating that the actual date used is different from the requested
    """
    idx = port_r.index
    if idx.empty:
        raise ValueError("Not enough return data to stress-test (portfolio returns empty).")

    '''
    If as_of_date is not provided, then assume the user is trying to stress test
    from the last cached date, such that this is just a pure forward simulation.
    In this case, there is no actual_realized_curve to return
    '''
    if not as_of_date_requested:
        applied_ts = idx[-1]
        return port_r.loc[:applied_ts], applied_ts.strftime("%Y-%m-%d"), None

    ts = pd.to_datetime(as_of_date_requested, errors="coerce")
    if pd.isna(ts):
        raise ValueError("as_of_date must be YYYY-MM-DD.")

    pos = idx.searchsorted(ts, side="left")
    if pos >= len(idx):
        raise ValueError(
            f"as_of_date is after the last date in the cached analysis window "
            f"({idx[-1].strftime('%Y-%m-%d')}). Re-run analysis with a later "
            f"end date to stress-test more recent data."
        )

    applied_ts = idx[pos]
    applied = applied_ts.strftime("%Y-%m-%d")
    note = None
    if applied != as_of_date_requested:
        note = f"Market closed on {as_of_date_requested}; using next trading day {applied}."

    return port_r.loc[:applied_ts], applied, note


def _build_stress_response(
    *,
    port_r: pd.Series,
    starting_cash: float,
    as_of_requested: str | None,
    as_of_applied: str,
    as_of_note: str | None,
    stoch_out: dict[str, Any],
    horizon: int,
    analysis_id: str,
    source: str,
    simulations: int,
    shock: dict[str, Any],
) -> dict[str, Any]:
    """
    Shared response shape for all stress transforms: 
        - historical_equity_curve: real equity curve up to the as-of date
        - actual_realized_curve: real equity curve from the as-of date to the end of the cached window 
          (i.e., the "what actually happened" baseline, empty if the as-of date is the last cached date)
        - forecase_paths: the simulated forecast bands from the as-of date forward.
        - terminal, drawdown, variance: distribution stats


    Parameters:
        - port_r: Portfolio returns series
        - starting_cash: The initial portfolio value at the beginning of the stress analysis window
        - as_of_requested: The requested date from which to begin the stress analysis
        - as_of_actual: The actual date used from which to begin the stress analysis
        - as_of_note: A message indicating whether the requested date was actually used
        - stoch_out: Dict containing the output of the simulation
        - horizon: Size of the stress analysis window in days
        - analysis_id: The type of stress transform used for analysis
        - source: Which cached return series the analysis was run against,
          "baseline" or "scenario" -- passed straight through into
          inputs.source, not used in any computation here
        - simulations: Number of Monte Carlo paths simulated (n_paths)
        - shock: Dict describing which shock mechanism produced stoch_out --
          e.g. {"type": "deterministic_regime", "drift_shift", "vol_mult"}
          or {"type": "calibrated_regime", "n_regimes", "selected_state",
          "regime_id"} -- passed straight through into inputs.shock

    Returns:
        - Dict representing the fully assembled response object
    """
    full_curve = equity_curve(port_r, starting_cash)
    as_of_ts = pd.to_datetime(as_of_applied)

    hist_curve = full_curve.loc[:as_of_ts]
    realized_curve = full_curve.loc[as_of_ts:].iloc[1:]  # exclude the as-of point itself

    future_idx = pd.bdate_range(as_of_ts + pd.Timedelta(days=1), periods=horizon)
    path_metrics = stoch_out["path_metrics"]
    terminal = stoch_out["terminal"]
    drawdown = stoch_out["drawdown"]

    resp: dict[str, Any] = {
        "inputs": {
            "analysis_id": analysis_id,
            "source": source,
            "as_of_date": {"requested": as_of_requested, "applied": as_of_applied, "note": as_of_note},
            "days": horizon,
            "simulations": simulations,
            "shock": shock,
        },
        "historical_equity_curve": serialize_series(hist_curve),
        "actual_realized_curve": serialize_series(realized_curve),
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
    }

    # Only calibrated_regime carries a variance path -- GBM (deterministic_regime)
    # has a constant sigma, so run_stochastic_forecast never puts "variance" in
    # stoch_out for it. No branching on shock type needed here; it just falls out.
    if "variance" in stoch_out:
        resp["variance"] = {
            k: float(v) for k, v in stoch_out["variance"].items() if k != "terminal_variances"
        }

    return resp


def _serialize_regime_stats(regime_stats: pd.DataFrame) -> list[dict[str, Any]]:
    """
    One entry per state, in state order, no auto-labeling,
    just the raw numbers for the user to read and decide from.
    
    Parameters:
        - regime_stats: DataFrame returned from the regime_engine model,
          contains the distribution parameters for each regime state

    Returns:
        - regime stats serialized as a python Dict
    """
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
    """
    Parameters: 
        - transition_probs: DataFrame returned from the regime_engine model,
          represents the transition probability matrix for each regime state

    Returns:
        - transition probs serialized as a python Dict
    
    """
    ordered = transition_probs.sort_index().sort_index(axis=1)
    return [[float(v) for v in row] for row in ordered.to_numpy()]

def run_deterministic_regime_stress_forecast(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Estimate real mu/sigma from cached returns up to an as-of date, shock
    them by user-provided drift_shift/vol_mult, and simulate forward via
    plain GBM Monte Carlo (run_stochastic_forecast(model="gbm", ...)).

    Parameters:
        - payload:
            - analysis_id (required): id from a prior /api/analyze call
            - source (optional, default "baseline"): "baseline" or "scenario"
            - as_of_date (optional, default: last cached date): YYYY-MM-DD,
              snapped forward to the next trading day if not one itself
            - days (optional, default 30): forecast horizon
            - simulations (optional, default 1000): number of Monte Carlo paths
            - drift_shift (optional, default -0.0005): added to daily mu
            - vol_mult (optional, default 1.5): multiplies daily sigma
            - random_seed (optional)

    Returns:
        - Dict representing the full response, built from _build_stress_response
          (historical_equity_curve, actual_realized_curve, forecast_paths,
          terminal, drawdown -- no variance, GBM has none)

    """
    analysis_id = str(payload.get("analysis_id", "")).strip()
    if not analysis_id:
        raise ValueError("analysis_id is required.")

    source = str(payload.get("source", "baseline")).strip().lower()
    if source not in ("baseline", "scenario"):
        raise ValueError("source must be 'baseline' or 'scenario'.")

    as_of_requested = payload.get("as_of_date")

    horizon = int(payload.get("days", 30))
    if horizon <= 0:
        raise ValueError("days must be > 0.")

    n_paths = int(payload.get("simulations", 1000))
    if n_paths <= 0:
        raise ValueError("simulations must be > 0.")

    drift_shift = float(payload.get("drift_shift", DETERMINISTIC_DRIFT_SHIFT_DEFAULT))
    vol_mult = float(payload.get("vol_mult", DETERMINISTIC_VOL_MULT_DEFAULT))
    if vol_mult <= 0:
        raise ValueError("vol_mult must be > 0.")

    random_seed = payload.get("random_seed")

    port_r, starting_cash, _cached_inputs = get_cached_returns_and_starting_cash(analysis_id, source)
    port_r = port_r.dropna()
    if port_r.empty:
        raise ValueError("Not enough return data to stress-test (portfolio returns empty).")

    port_r_hist, as_of_applied, as_of_note = _truncate_returns(port_r, as_of_requested)

    mu_daily, _ = estimate_drift(port_r_hist, "mean")
    sigma_daily, _ = estimate_volatility(port_r_hist, "historical")

    mu_shocked_daily = mu_daily + drift_shift
    sigma_shocked_daily = sigma_daily * vol_mult

    s0 = float(equity_curve(port_r_hist, starting_cash).iloc[-1])
    T = horizon / TRADING_DAYS_PER_YEAR

    stoch_out = run_stochastic_forecast(
        model="gbm",
        s0=s0,
        mu=mu_shocked_daily * TRADING_DAYS_PER_YEAR,
        sigma=sigma_shocked_daily * math.sqrt(TRADING_DAYS_PER_YEAR),
        T=T,
        N=horizon,
        n=n_paths,
        random_seed=random_seed,
    )

    return _build_stress_response(
        port_r=port_r,
        starting_cash=starting_cash,
        as_of_requested=as_of_requested,
        as_of_applied=as_of_applied,
        as_of_note=as_of_note,
        stoch_out=stoch_out,
        horizon=horizon,
        analysis_id=analysis_id,
        source=source,
        simulations=n_paths,
        shock={
            "type": "deterministic_regime",
            "drift_shift": drift_shift,
            "vol_mult": vol_mult,
        },
    )


def preview_calibrated_regime_states(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Fit a regime-switching GMM on cached returns truncated to an as-of date.
    Returns each state's stats for the user to inspect, and caches what
    step 2 needs (including the full, untruncated returns, so step 2 can
    build the actual_realized_curve baseline without refitting a model).

    Parameters:
        - payload:
            - analysis_id (required): id from a prior /api/analyze call
            - source (optional, default "baseline"): "baseline" or "scenario"
            - n_regimes (optional, default 3, must be 2-8): number of GMM states to fit
            - as_of_date (optional, default: last cached date): YYYY-MM-DD,
              snapped forward to the next trading day if not one itself

    Returns:
        - Dict containing regime_id (pass to run_calibrated_regime_stress_test),
          n_regimes, as_of_date, regime_stats (per-state mean/std/annualized
          return/vol, no auto-labeling), transition_probs (n_regimes x n_regimes
          nested list), current_regime_probs (nowcast as of as_of_date)
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

    as_of_requested = payload.get("as_of_date")

    port_r, starting_cash, _cached_inputs = get_cached_returns_and_starting_cash(analysis_id, source)
    port_r = port_r.dropna()
    if port_r.empty:
        raise ValueError("Not enough return data to fit regimes (portfolio returns empty).")

    port_r_hist, as_of_applied, as_of_note = _truncate_returns(port_r, as_of_requested)

    params = calibrate_regime_params(port_r=port_r_hist, n_regimes=n_regimes)

    regime_id = analysis_store.put({
        "kind": "calibrated_regime_preview",
        "source_analysis_id": analysis_id,
        "source": source,
        "n_regimes": n_regimes,
        "regime_stats": params.regime_stats,
        "transition_probs": params.transition_probs,
        "port_r": port_r,                 # full, untruncated
        "starting_cash": starting_cash,
        "as_of_requested": as_of_requested,
        "as_of_applied": as_of_applied,
        "as_of_note": as_of_note,
    })

    return {
        "regime_id": regime_id,
        "n_regimes": n_regimes,
        "as_of_date": {"requested": as_of_requested, "applied": as_of_applied, "note": as_of_note},
        "regime_stats": _serialize_regime_stats(params.regime_stats),
        "transition_probs": _serialize_transition_probs(params.transition_probs),
        "current_regime_probs": [float(p) for p in params.current_regime_probs],
    }


def run_calibrated_regime_stress_test(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Force every simulated path to start in a user-selected regime state
    (from a prior preview_calibrated_regime_states call) and simulate
    forward via run_stochastic_forecast(model="regime", ...). Uses the
    cached fit -- no refitting.

    Parameters:
        - payload:
            - regime_id (required): from a prior preview_calibrated_regime_states call
            - selected_state (required): 0..n_regimes-1, which fitted state to force every path into
            - days (optional, default 30): forecast horizon
            - simulations (optional, default 1000): number of Monte Carlo paths
            - random_seed (optional)

    Returns:
        - Dict representing the full response, built from _build_stress_response
          (historical_equity_curve, actual_realized_curve, forecast_paths,
          terminal, drawdown, variance)
    """
    regime_id = str(payload.get("regime_id", "")).strip()
    if not regime_id:
        raise ValueError("regime_id is required. Call the calibrated-regime preview endpoint first.")

    cached = analysis_store.get(regime_id)
    if cached is None:
        raise ValueError("regime_id not found or expired. Re-run the calibrated-regime preview.")
    if cached.get("kind") != "calibrated_regime_preview":
        raise ValueError("regime_id does not refer to a calibrated-regime preview.")

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

    port_r: pd.Series = cached["port_r"]
    starting_cash: float = cached["starting_cash"]
    as_of_applied: str = cached["as_of_applied"]

    s0 = float(equity_curve(port_r.loc[:as_of_applied], starting_cash).iloc[-1])
    T = horizon / TRADING_DAYS_PER_YEAR

    start_regime_probs = np.zeros(n_regimes)
    start_regime_probs[selected_state] = 1.0

    # s0 on this dataclass is unused by run_stochastic_forecast's "regime"
    # branch (it takes s0 as its own top-level kwarg) -- kept here only for
    # metadata completeness.
    regime_params = RegimeCalibratedParams(
        portfolio_name=str(cached.get("source_analysis_id", "portfolio")),
        start=port_r.index[0].strftime("%Y-%m-%d"),
        end=as_of_applied,
        n_regimes=n_regimes,
        s0=s0,
        regime_stats=cached["regime_stats"],
        transition_probs=cached["transition_probs"],
        current_regime_probs=start_regime_probs,
    )

    stoch_out = run_stochastic_forecast(
        model="regime",
        s0=s0,
        T=T,
        N=horizon,
        n=n_paths,
        regime_params=regime_params,
        random_seed=random_seed,
    )

    return _build_stress_response(
        port_r=port_r,
        starting_cash=starting_cash,
        as_of_requested=cached.get("as_of_requested"),
        as_of_applied=as_of_applied,
        as_of_note=cached.get("as_of_note"),
        stoch_out=stoch_out,
        horizon=horizon,
        analysis_id=cached.get("source_analysis_id"),
        source=cached.get("source"),
        simulations=n_paths,
        shock={
            "type": "calibrated_regime",
            "n_regimes": n_regimes,
            "selected_state": selected_state,
            "regime_id": regime_id,
        },
    )
