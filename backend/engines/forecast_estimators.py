from typing import Any
import numpy as np
import pandas as pd


DEFAULT_ROLLING_WINDOW = 60
DEFAULT_LAMBDA = 0.94
DEFAULT_ALPHA = 1.0 - DEFAULT_LAMBDA


def estimate_drift(
    port_r: pd.Series,
    mode: str,
    *,
    window: int | None = None,
    alpha: float | None = None,
    lam: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Estimate drift (mean return) from portfolio returns.

    Returns:
      r_hat: float daily drift estimate
      trend_meta: dict metadata for response
    """
    mode = (mode or "").strip().lower()
    s = port_r.dropna()

    if s.empty:
        raise ValueError("Not enough return data to forecast (portfolio returns empty).")

    if mode == "mean":
        r_hat = float(s.mean())
        return r_hat, {"mode": "mean", "mean_daily_return": r_hat}

    if mode == "rolling":
        if window is None:
            window = DEFAULT_ROLLING_WINDOW

        try:
            window = int(window)
        except Exception:
            raise ValueError("forecast.window must be an integer for mode='rolling'.")

        if window <= 0:
            raise ValueError("forecast.window must be > 0 for mode='rolling'.")

        if len(s) < window:
            raise ValueError(
                f"Not enough return data for rolling window (need {window}, have {len(s)})."
            )

        r_hat = float(s.iloc[-window:].mean())
        return r_hat, {"mode": "rolling", "window": window, "mean_daily_return": r_hat}

    if mode == "ewma":
        if alpha is None and lam is None:
            lam = DEFAULT_LAMBDA
            alpha = DEFAULT_ALPHA

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

    raise ValueError("drift mode must be 'mean', 'rolling', or 'ewma'.")


def estimate_volatility(
    port_r: pd.Series,
    mode: str,
    *,
    window: int | None = None,
    alpha: float | None = None,
    lam: float | None = None,
    ddof: int = 1,
) -> tuple[float, dict[str, Any]]:
    """
    Estimate volatility (standard deviation of returns) from portfolio returns.

    Returns:
      sigma_hat: float daily volatility estimate
      vol_meta: dict metadata for response
    """
    mode = (mode or "").strip().lower()
    s = port_r.dropna()

    if s.empty:
        raise ValueError("Not enough return data to estimate volatility (portfolio returns empty).")

    if mode == "historical":
        if len(s) < 2:
            raise ValueError("Need at least 2 returns to estimate historical volatility.")

        sigma_hat = float(s.std(ddof=ddof))
        return sigma_hat, {
            "mode": "historical",
            "daily_volatility": sigma_hat,
            "annualized_volatility": float(sigma_hat * np.sqrt(252)),
            "ddof": int(ddof),
        }

    if mode == "rolling":
        if window is None:
            window = DEFAULT_ROLLING_WINDOW

        try:
            window = int(window)
        except Exception:
            raise ValueError("forecast.window must be an integer for mode='rolling'.")

        if window <= 1:
            raise ValueError("forecast.window must be > 1 for mode='rolling'.")

        if len(s) < window:
            raise ValueError(
                f"Not enough return data for rolling window (need {window}, have {len(s)})."
            )

        sigma_hat = float(s.iloc[-window:].std(ddof=ddof))
        return sigma_hat, {
            "mode": "rolling",
            "window": window,
            "daily_volatility": sigma_hat,
            "annualized_volatility": float(sigma_hat * np.sqrt(252)),
            "ddof": int(ddof),
        }

    if mode == "ewma":
        if alpha is None and lam is None:
            lam = DEFAULT_LAMBDA
            alpha = DEFAULT_ALPHA

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

        if len(s) < 2:
            raise ValueError("Need at least 2 returns to estimate EWMA volatility.")

        mean_r = float(s.mean())
        centered = s - mean_r

        # Seed variance with sample variance
        var = float(centered.var(ddof=ddof))
        one_minus_lambda = 1.0 - float(lam)

        for r in centered.iloc[1:]:
            var = float(lam) * var + one_minus_lambda * float(r) ** 2

        sigma_hat = float(np.sqrt(var))
        return sigma_hat, {
            "mode": "ewma",
            "alpha": float(alpha),
            "lambda": float(lam),
            "daily_volatility": sigma_hat,
            "annualized_volatility": float(sigma_hat * np.sqrt(252)),
            "ddof": int(ddof),
        }

    raise ValueError("volatility mode must be 'historical', 'rolling', or 'ewma'.")