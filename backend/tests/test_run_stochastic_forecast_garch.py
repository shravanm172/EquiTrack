"""
Verification tests for calibrate_garch_mle + the GARCH path through
run_stochastic_forecast -- does the code compute what it's supposed to,
given known synthetic inputs. Not a validation test (is GARCH a good
forecast); that's what the real-data backtests are for.
"""
import pytest

from engines.garch_engine import calibrate_garch_mle
from engines.stochastic_engine import run_stochastic_forecast


def test_calibrate_garch_mle_stationarity_constraint_holds(synthetic_returns):
    """
    Real invariant, not a golden value: alpha+beta must stay below 1 for the
    model to be covariance-stationary (mean-reverting, finite long-run
    variance) -- enforced explicitly during MLE optimization.
    """
    params = calibrate_garch_mle(returns=synthetic_returns, estimate_mu=False)

    assert params.alpha + params.beta < 1.0
    assert params.alpha >= 0.0
    assert params.beta >= 0.0
    assert params.omega > 0.0
    assert params.nu > 2.0


def test_calibrate_garch_mle_matches_golden_reference(synthetic_returns):
    """
    Golden-value regression check: GARCH's MLE fit has no closed form, so
    there's no way to independently hand-derive the "correct" answer --
    these values were computed once from the real code and are treated as
    the trusted reference. If this ever changes, the calibration behavior
    changed, intentionally or not.
    """
    params = calibrate_garch_mle(returns=synthetic_returns, estimate_mu=False)

    assert params.mu == pytest.approx(0.0)
    assert params.omega == pytest.approx(2.481780671622555e-06, rel=1e-6)
    assert params.alpha == pytest.approx(0.10517958822199343, rel=1e-6)
    assert params.beta == pytest.approx(0.8756462504346406, rel=1e-6)
    assert params.nu == pytest.approx(13.01260247500445, rel=1e-6)
    assert params.h0 == pytest.approx(0.00017329194640780017, rel=1e-6)


def test_run_stochastic_forecast_garch_output_contract(synthetic_returns):
    params = calibrate_garch_mle(returns=synthetic_returns, estimate_mu=False)

    out = run_stochastic_forecast(
        model="garch", s0=100.0, T=30 / 252, N=30, n=500,
        garch_params={
            "mu": params.mu, "omega": params.omega, "alpha": params.alpha,
            "beta": params.beta, "h0": params.h0, "nu": params.nu,
        },
        random_seed=123,
    )

    assert out["model"] == "garch"
    assert out["paths"].shape == (500, 31)
    assert out["variance_paths"].shape == (500, 31)
    assert "variance" in out

    for key in ("terminal", "drawdown", "path_metrics"):
        assert key in out


def test_run_stochastic_forecast_garch_missing_params_raises():
    with pytest.raises(ValueError, match="garch_params is required"):
        run_stochastic_forecast(model="garch", s0=100.0, T=1.0, N=10, n=10)


def test_run_stochastic_forecast_garch_reproducible_with_fixed_seed(synthetic_returns):
    params = calibrate_garch_mle(returns=synthetic_returns, estimate_mu=False)

    out = run_stochastic_forecast(
        model="garch", s0=100.0, T=30 / 252, N=30, n=500,
        garch_params={
            "mu": params.mu, "omega": params.omega, "alpha": params.alpha,
            "beta": params.beta, "h0": params.h0, "nu": params.nu,
        },
        random_seed=123,
    )

    assert out["terminal"]["mean_terminal_value"] == pytest.approx(100.20166110323494, rel=1e-6)
