"""
Verification tests for the GBM path through run_stochastic_forecast --
does the code compute what it's supposed to, given known synthetic inputs.
Not a validation test (is GBM a good forecast); that's what the real-data
backtests are for. Low-level simulate_gbm_path/simulate_many_paths and the
shared summarize_* helpers already have thorough coverage in
test_stochastic_forecast_engine.py -- this file covers the orchestration
layer (run_stochastic_forecast itself) that sits on top of them.
"""
import numpy as np
import pytest

from engines.forecast_estimators import estimate_drift, estimate_volatility
from engines.stochastic_engine import run_stochastic_forecast

TRADING_DAYS_PER_YEAR = 252


def test_estimate_drift_and_volatility_match_closed_form(synthetic_returns):
    """
    GBM's calibration is genuinely closed-form -- no golden value needed,
    just check it against a direct mean/std computation.
    """
    mu_daily, _ = estimate_drift(synthetic_returns, "mean")
    sigma_daily, _ = estimate_volatility(synthetic_returns, "historical")

    assert mu_daily == pytest.approx(synthetic_returns.mean())
    assert sigma_daily == pytest.approx(synthetic_returns.std(ddof=1))


def test_run_stochastic_forecast_gbm_output_contract(synthetic_returns):
    mu_daily, _ = estimate_drift(synthetic_returns, "mean")
    sigma_daily, _ = estimate_volatility(synthetic_returns, "historical")
    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    out = run_stochastic_forecast(
        model="gbm", s0=100.0, mu=mu_annual, sigma=sigma_annual,
        T=30 / TRADING_DAYS_PER_YEAR, N=30, n=500, random_seed=123,
    )

    assert out["model"] == "gbm"
    assert out["paths"].shape == (500, 31)
    assert out["variance_paths"] is None
    assert "variance" not in out  # only added when variance_paths is not None
    assert out["params"]["mu"] == pytest.approx(mu_annual)
    assert out["params"]["sigma"] == pytest.approx(sigma_annual)

    for key in ("terminal", "drawdown", "path_metrics"):
        assert key in out


def test_run_stochastic_forecast_gbm_missing_sigma_raises():
    with pytest.raises(ValueError, match="sigma is required"):
        run_stochastic_forecast(model="gbm", s0=100.0, mu=0.05, T=1.0, N=10, n=10)


def test_run_stochastic_forecast_gbm_missing_mu_raises():
    with pytest.raises(ValueError, match="mu is required"):
        run_stochastic_forecast(model="gbm", s0=100.0, sigma=0.2, T=1.0, N=10, n=10)


def test_run_stochastic_forecast_gbm_reproducible_with_fixed_seed(synthetic_returns):
    """
    Golden-value regression check: with a fixed seed, run_stochastic_forecast
    must reproduce exactly the same paths every time -- computed once from
    the real code and hardcoded here. If this ever changes, something in
    the pipeline changed behavior, intentionally or not.
    """
    mu_daily, _ = estimate_drift(synthetic_returns, "mean")
    sigma_daily, _ = estimate_volatility(synthetic_returns, "historical")
    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    out = run_stochastic_forecast(
        model="gbm", s0=100.0, mu=mu_annual, sigma=sigma_annual,
        T=30 / TRADING_DAYS_PER_YEAR, N=30, n=500, random_seed=123,
    )

    assert out["terminal"]["mean_terminal_value"] == pytest.approx(100.04355745800716, rel=1e-6)
    assert np.allclose(out["paths"][0, :3], [100.0, 98.68853197, 98.19414462], atol=1e-6)
