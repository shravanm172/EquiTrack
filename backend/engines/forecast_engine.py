from typing import Any
import pandas as pd

from services.store_singleton import analysis_store
from engines.analytics_engine import equity_curve
from engines.analytics_engine import forecast_summary


DEFAULT_ROLLING_WINDOW = 60
DEFAULT_LAMBDA = 0.94
DEFAULT_ALPHA = 1.0 - DEFAULT_LAMBDA

def _estimate_drift(
    port_r: pd.Series,
    mode: str,
    *,
    window: int | None = None,
    alpha: float | None = None,
    lam: float | None = None,   
) -> tuple[float, dict[str, Any]]:
    """
    Estimate drift (mean return) from portfolio returns using specified method.

    Returns:
      r_hat: float drift estimate (daily)
      trend_meta: dict to include in response["trend"]
    """
    mode = (mode or "").strip().lower()

    if mode == "mean": # Simple historical mean: use the mean of ALL available returns
        r_hat = float(port_r.mean())
        return r_hat, {"mode": "mean", "mean_daily_return": r_hat}

    if mode == "rolling":
        if window is None:
            window = DEFAULT_ROLLING_WINDOW  # default

        try:
            window = int(window)
        except Exception:
            raise ValueError("forecast.window must be an integer for mode='rolling'.")

        if window <= 0:
            raise ValueError("forecast.window must be > 0 for mode='rolling'.")

        if len(port_r) < window:
            raise ValueError(
                f"Not enough return data for rolling window (need {window}, have {len(port_r)})."
            )

        r_hat = float(port_r.iloc[-window:].mean())
        return r_hat, {"mode": "rolling", "window": window, "mean_daily_return": r_hat}

    if mode == "ewma":
        # Allow either alpha or lambda
        # RiskMetrics daily default lambda=0.94 => alpha=0.06
        if alpha is None and lam is None:
            lam = DEFAULT_LAMBDA
            alpha = DEFAULT_ALPHA

        # If alpha provided, it wins; otherwise derive from lambda
        if alpha is not None:
            try:
                alpha = float(alpha)
            except Exception:
                raise ValueError("forecast.alpha must be a number for mode='ewma'.")
            if not (0.0 < alpha < 1.0):
                raise ValueError("forecast.alpha must be in (0, 1) for mode='ewma'.")
            lam = 1.0 - alpha
        else:
            try:
                lam = float(lam)
            except Exception:
                raise ValueError("forecast.lambda must be a number for mode='ewma'.")
            if not (0.0 < lam < 1.0):
                raise ValueError("forecast.lambda must be in (0, 1) for mode='ewma'.")
            alpha = 1.0 - lam

        # EWMA recursion: mu_t = alpha*r_t + (1-alpha)*mu_{t-1}
        # Seed mu_0 as the first return.
        s = port_r.dropna()
        if s.empty:
            raise ValueError("Not enough return data to forecast (portfolio returns empty).")

        mu = float(s.iloc[0])
        one_minus_alpha = 1.0 - float(alpha)
        for r in s.iloc[1:]:
            mu = float(alpha) * float(r) + one_minus_alpha * mu

        r_hat = float(mu)
        return r_hat, {
            "mode": "ewma",
            "alpha": float(alpha),
            "lambda": float(lam),
            "mean_daily_return": r_hat,
        }

    raise ValueError("forecast.mode must be 'mean', 'rolling', or 'ewma'.")



def _forecast_from_returns(
    port_r: pd.Series,
    starting_cash: float,
    forecast_days: int,
    *,
    mode: str = "mean",
    window: int | None = None,
    alpha: float | None = None,
    lam: float | None = None,
) -> dict[str, Any]:
    """
    Generate forecast equity curve from portfolio returns and specified drift estimation method.

    Returns combined historical + forecast equity curve and summary stats.
    """

    if forecast_days <= 0:
        raise ValueError("forecast_days must be > 0")

    port_r = port_r.dropna()
    if port_r.empty:
        raise ValueError("Not enough return data to forecast (portfolio returns empty).")

    hist_curve = equity_curve(port_r, starting_cash)
    if not isinstance(hist_curve, pd.Series):
        raise TypeError("equity_curve must return a pandas Series.")

    r_hat, trend_meta = _estimate_drift(port_r, mode, window=window, alpha=alpha, lam=lam) # Estimate drift using specified method

    last_date = hist_curve.index[-1]
    future_idx = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

    cur = float(hist_curve.iloc[-1])
    vals = []
    for _ in range(forecast_days):
        cur *= (1.0 + r_hat)
        vals.append(cur)

    forecast_curve = pd.Series(vals, index=future_idx)
    combined = pd.concat([hist_curve, forecast_curve])

    hist_json = [
        {"date": i.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for i, v in hist_curve.items()
    ]
    fc_json = [
        {"date": i.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for i, v in forecast_curve.items()
    ]
    combined_json = [
        {"date": i.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for i, v in combined.items()
    ]

    summary = forecast_summary(
        hist_curve=hist_curve,
        forecast_curve=forecast_curve,
        trend=trend_meta,          
        target_multiple=1.10,      
    )

    return {
        "trend": trend_meta,
        "summary": summary,  
        "historical_equity_curve": hist_json,
        "forecast_equity_curve": fc_json,
        "equity_curve": combined_json,  # combined for plotting
    }


def forecast_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Main entry point for portfolio forecasting.  

    Returns forecasted equity curve and summary stats based on historical portfolio returns and specified forecasting method.
    """
    analysis_id = str(payload.get("analysis_id", "")).strip()
    if not analysis_id:
        raise ValueError("analysis_id is required.")

    forecast_cfg = payload.get("forecast", {}) or {}
    forecast_days = int(forecast_cfg.get("days", 30))
    mode = str(forecast_cfg.get("mode", "mean")).strip().lower()

    # rolling-specific param
    window = forecast_cfg.get("window", None)

    # ewma specific params
    alpha = forecast_cfg.get("alpha", None)
    lam = forecast_cfg.get("lambda", None)

    # For shock analyses, user chooses baseline vs scenario
    source = str(payload.get("source", "baseline")).strip().lower()
    if source not in ("baseline", "scenario"):
        raise ValueError("source must be 'baseline' or 'scenario'.")

    item = analysis_store.get(analysis_id)
    if item is None:
        raise ValueError("analysis_id not found or expired. Re-run analysis.")

    kind = item.get("kind")

    if kind == "analyze":
        port_r = item["portfolio_returns"]
        starting_cash = float(item["inputs"]["starting_cash"])

    elif kind == "analyze_shock":
        port_r = item["baseline_returns"] if source == "baseline" else item["scenario_returns"]
        starting_cash = float(item["inputs"]["starting_cash"])

    else:
        raise ValueError(f"Unsupported cached analysis kind: {kind}")

    forecast_out = _forecast_from_returns(
        port_r,
        starting_cash,
        forecast_days,
        mode=mode,
        window=window,
        alpha=alpha,
        lam=lam,
    )

    # echo back what was used
    inputs_forecast = {"days": forecast_days, "mode": mode}
    if mode == "rolling" and window is not None:
        inputs_forecast["window"] = int(window)
    if mode == "ewma":
        if alpha is not None: # prefers alpha
            inputs_forecast["alpha"] = float(alpha)
        elif lam is not None:
            inputs_forecast["lambda"] = float(lam)
        else:
            inputs_forecast["lambda"] = 0.94  # default

    return {
        "inputs": {
            "analysis_id": analysis_id,
            "source": source,
            "forecast": inputs_forecast,
        },
        **forecast_out,
    }
