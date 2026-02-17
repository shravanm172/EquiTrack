from __future__ import annotations
import math
from typing import Any

import pandas as pd

# Small math helpers
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _clamp_pct(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return x

# ================Historical analysis metrics======================

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


# ====================Forecasting analysis metrics======================
def forecast_final_value(forecast_curve: pd.Series) -> float:
    """Last value of forecast window."""
    if forecast_curve is None or forecast_curve.empty:
        return 0.0
    return _safe_float(forecast_curve.iloc[-1], 0.0)


def forecast_total_return(hist_curve: pd.Series, forecast_curve: pd.Series) -> float:
    """
    Forecast-window total return relative to the last historical value:
      (V_end_forecast / V_last_hist) - 1
    """
    if hist_curve is None or hist_curve.empty:
        return 0.0
    if forecast_curve is None or forecast_curve.empty:
        return 0.0

    v0 = _safe_float(hist_curve.iloc[-1], 0.0)
    v1 = _safe_float(forecast_curve.iloc[-1], 0.0)
    if v0 <= 0:
        return 0.0
    return _clamp_pct((v1 / v0) - 1.0)


def forecast_absolute_change(hist_curve: pd.Series, forecast_curve: pd.Series) -> float:
    """Absolute change over forecast window: V_end_forecast - V_last_hist."""
    if hist_curve is None or hist_curve.empty:
        return 0.0
    if forecast_curve is None or forecast_curve.empty:
        return 0.0

    v0 = _safe_float(hist_curve.iloc[-1], 0.0)
    v1 = _safe_float(forecast_curve.iloc[-1], 0.0)
    return v1 - v0


def forecast_avg_daily_return_from_equity(forecast_curve: pd.Series) -> float:
    """
    Average daily return implied by the forecast equity curve:
      mean( V_t / V_{t-1} - 1 )
    (This will equal r_hat for your constant-drift model, but it’s still a nice sanity metric.)
    """
    if forecast_curve is None or len(forecast_curve) < 2:
        return 0.0

    s = forecast_curve.astype("float64")
    rets = s.pct_change().dropna()
    if rets.empty:
        return 0.0
    return _safe_float(rets.mean(), 0.0)


def forecast_days_to_target(
    hist_curve: pd.Series,
    forecast_curve: pd.Series,
    target_multiple: float = 1.10,
) -> int | None:
    """
    Days in forecast window to reach target_multiple * last_hist_value.
    Example: target_multiple=1.10 => +10% target.
    Returns:
      - integer day count (1-based within forecast window)
      - None if not reached or inputs invalid
    """
    if hist_curve is None or hist_curve.empty:
        return None
    if forecast_curve is None or forecast_curve.empty:
        return None

    v0 = _safe_float(hist_curve.iloc[-1], 0.0)
    if v0 <= 0:
        return None

    if target_multiple <= 0:
        return None

    target = v0 * float(target_multiple)
    eps = 1e-12
    vals = forecast_curve.astype("float64").values
    
    for i, v in enumerate(vals, start = 1):
        if float(v) + eps >= target:
            return i  # 1-based day index
    return None


def forecast_summary(
    *,
    hist_curve: pd.Series,
    forecast_curve: pd.Series,
    trend: dict[str, Any] | None = None,
    target_multiple: float = 1.10,
) -> dict[str, Any]:
    """
    One-stop summary for a single forecast run.
    Intended to be returned by /api/forecast, then deltas computed in frontend.
    """
    v_last_hist = _safe_float(hist_curve.iloc[-1], 0.0) if hist_curve is not None and not hist_curve.empty else 0.0
    v_end_fc = forecast_final_value(forecast_curve)

    out: dict[str, Any] = {
        "last_historical_value": round(v_last_hist, 2),
        "forecast_end_value": round(v_end_fc, 2),
        "forecast_abs_change": round(forecast_absolute_change(hist_curve, forecast_curve), 2),
        "forecast_total_return": round(forecast_total_return(hist_curve, forecast_curve), 6),
        "forecast_avg_daily_return": round(forecast_avg_daily_return_from_equity(forecast_curve), 6),
        "days_to_target_multiple": forecast_days_to_target(hist_curve, forecast_curve, target_multiple=target_multiple),
        "target_multiple": float(target_multiple),
    }

    # Optional: include the drift estimate you used, so UI can display it
    if trend:
        out["trend"] = trend

    return out