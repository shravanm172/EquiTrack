'''
parse payload

validate request

load cached analysis data

choose baseline vs scenario

decide forecast type

call the appropriate engine
'''

from __future__ import annotations

from typing import Any
import math
import pandas as pd

from services.store_singleton import analysis_store
from engines.analytics_engine import equity_curve
from engines.forecast_engine import _forecast_from_returns
from engines.forecast_estimators import estimate_drift, estimate_volatility
from engines.stochastic_engine import run_stochastic_forecast


TRADING_DAYS_PER_YEAR = 252


def _get_cached_returns_and_starting_cash(
    analysis_id: str,
    source: str,
) -> tuple[pd.Series, float]:
    """
    Load cached return series and starting cash from analysis_store.
    """
    item = analysis_store.get(analysis_id)
    if item is None:
        raise ValueError("analysis_id not found or expired. Re-run analysis.")

    kind = item.get("kind")

    if kind == "analyze":
        if source != "baseline":
            raise ValueError("source must be 'baseline' for non-shock analyses.")
        port_r = item["portfolio_returns"]
        starting_cash = float(item["inputs"]["starting_cash"])
        return port_r, starting_cash

    if kind == "analyze_shock":
        port_r = item["baseline_returns"] if source == "baseline" else item["scenario_returns"]
        starting_cash = float(item["inputs"]["starting_cash"])
        return port_r, starting_cash

    raise ValueError(f"Unsupported cached analysis kind: {kind}")


def _serialize_series(series: pd.Series) -> list[dict[str, Any]]:
    return [
        {"date": idx.strftime("%Y-%m-%d"), "value": round(float(val), 2)}
        for idx, val in series.items()
    ]


def _serialize_band_series(index: pd.Index, values) -> list[dict[str, Any]]:
    return [
        {"date": idx.strftime("%Y-%m-%d"), "value": round(float(val), 2)}
        for idx, val in zip(index, values)
    ]


def _run_deterministic_forecast(
    port_r: pd.Series,
    starting_cash: float,
    forecast_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic forecast branch.
    """
    forecast_days = int(forecast_cfg.get("days", 30))
    drift_mode = str(forecast_cfg.get("drift_mode", forecast_cfg.get("mode", "mean"))).strip().lower()

    window = forecast_cfg.get("window", None)
    alpha = forecast_cfg.get("alpha", None)
    lam = forecast_cfg.get("lambda", None)

    out = _forecast_from_returns(
        port_r=port_r,
        starting_cash=starting_cash,
        forecast_days=forecast_days,
        mode=drift_mode,
        window=window,
        alpha=alpha,
        lam=lam,
    )

    inputs_forecast = {
        "type": "deterministic",
        "days": forecast_days,
        "drift_mode": drift_mode,
    }

    if drift_mode == "rolling" and window is not None:
        inputs_forecast["window"] = int(window)

    if drift_mode == "ewma":
        if alpha is not None:
            inputs_forecast["alpha"] = float(alpha)
        elif lam is not None:
            inputs_forecast["lambda"] = float(lam)
        else:
            inputs_forecast["lambda"] = 0.94

    return {
        "inputs_forecast": inputs_forecast,
        **out,
    }


def _run_stochastic_forecast(
    port_r: pd.Series,
    starting_cash: float,
    forecast_cfg: dict[str, Any],
) -> dict[str, Any]:
    forecast_days = int(forecast_cfg.get("days", 30))
    if forecast_days <= 0:
        raise ValueError("forecast.days must be > 0.")

    simulations = int(forecast_cfg.get("simulations", 1000))
    if simulations <= 0:
        raise ValueError("forecast.simulations must be > 0.")

    drift_mode = str(forecast_cfg.get("drift_mode", "mean")).strip().lower()
    vol_mode = str(forecast_cfg.get("vol_mode", "historical")).strip().lower()

    window = forecast_cfg.get("window", None)
    alpha = forecast_cfg.get("alpha", None)
    lam = forecast_cfg.get("lambda", None)

    port_r = port_r.dropna()
    if port_r.empty:
        raise ValueError("Not enough return data to forecast (portfolio returns empty).")

    hist_curve = equity_curve(port_r, starting_cash)
    if not isinstance(hist_curve, pd.Series):
        raise TypeError("equity_curve must return a pandas Series.")

    mu_daily, trend_meta = estimate_drift(
        port_r,
        drift_mode,
        window=window,
        alpha=alpha,
        lam=lam,
    )
    sigma_daily, vol_meta = estimate_volatility(
        port_r,
        vol_mode,
        window=window,
        alpha=alpha,
        lam=lam,
    )

    mu_annual = float(mu_daily) * TRADING_DAYS_PER_YEAR
    sigma_annual = float(sigma_daily) * math.sqrt(TRADING_DAYS_PER_YEAR)

    s0 = float(hist_curve.iloc[-1])
    T = forecast_days / TRADING_DAYS_PER_YEAR
    N = forecast_days

    stoch_out = run_stochastic_forecast(
        s0=s0,
        mu=mu_annual,
        sigma=sigma_annual,
        T=T,
        N=N,
        n=simulations,
    )

    last_date = hist_curve.index[-1]
    future_idx = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

    path_metrics = stoch_out["path_metrics"]

    forecast_paths = {
        "p10": _serialize_band_series(future_idx, path_metrics["p10_path"][1:]),
        "p25": _serialize_band_series(future_idx, path_metrics["p25_path"][1:]),
        "p50": _serialize_band_series(future_idx, path_metrics["p50_path"][1:]),
        "p75": _serialize_band_series(future_idx, path_metrics["p75_path"][1:]),
        "p90": _serialize_band_series(future_idx, path_metrics["p90_path"][1:]),
    }

    terminal = stoch_out["terminal"]
    drawdown = stoch_out["drawdown"]

    inputs_forecast = {
        "type": "stochastic",
        "days": forecast_days,
        "simulations": simulations,
        "drift_mode": drift_mode,
        "vol_mode": vol_mode,
    }

    if window is not None and (drift_mode == "rolling" or vol_mode == "rolling"):
        inputs_forecast["window"] = int(window)

    if drift_mode == "ewma" or vol_mode == "ewma":
        if alpha is not None:
            inputs_forecast["alpha"] = float(alpha)
        elif lam is not None:
            inputs_forecast["lambda"] = float(lam)
        else:
            inputs_forecast["lambda"] = 0.94

    return {
        "inputs_forecast": inputs_forecast,
        "trend": {
            **trend_meta,
            "annualized_drift": float(mu_annual),
        },
        "volatility": {
            **vol_meta,
            "daily_volatility": float(sigma_daily),
            "annualized_volatility": float(sigma_annual),
        },
        "historical_equity_curve": _serialize_series(hist_curve),
        "forecast_paths": forecast_paths,
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


def forecast_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Single forecast service entrypoint.

    Payload styles:

    Deterministic:
    {
      "analysis_id": "...",
      "source": "baseline",
      "forecast": {
        "type": "deterministic",
        "days": 30,
        "drift_mode": "mean"
      }
    }

    Stochastic:
    {
      "analysis_id": "...",
      "source": "scenario",
      "forecast": {
        "type": "stochastic",
        "days": 252,
        "simulations": 10000,
        "drift_mode": "mean",
        "vol_mode": "historical"
      }
    }
    """
    analysis_id = str(payload.get("analysis_id", "")).strip()
    if not analysis_id:
        raise ValueError("analysis_id is required.")

    source = str(payload.get("source", "baseline")).strip().lower()
    if source not in ("baseline", "scenario"):
        raise ValueError("source must be 'baseline' or 'scenario'.")

    forecast_cfg = payload.get("forecast", {}) or {}
    forecast_type = str(forecast_cfg.get("type", "deterministic")).strip().lower()
    if forecast_type not in ("deterministic", "stochastic"):
        raise ValueError("forecast.type must be 'deterministic' or 'stochastic'.")

    port_r, starting_cash = _get_cached_returns_and_starting_cash(analysis_id, source)

    if forecast_type == "deterministic":
        out = _run_deterministic_forecast(port_r, starting_cash, forecast_cfg)
    else:
        out = _run_stochastic_forecast(port_r, starting_cash, forecast_cfg)

    return {
        "inputs": {
            "analysis_id": analysis_id,
            "source": source,
            "forecast": out.pop("inputs_forecast"),
        },
        **out,
    }