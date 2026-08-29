"""
Verification tests for calibrate_regime_params + simulate_regime_switching_paths
+ the regime path through run_stochastic_forecast -- does the code compute
what it's supposed to, given known synthetic inputs. Not a validation test
(is the regime-switching model a good forecast); that's what the real-data
backtest in engines/ml/rolling_backtest_forecasting_models.py is for.
"""
import numpy as np
import pandas as pd
import pytest

from engines.regime_engine import (
    calibrate_regime_params,
    simulate_regime_switching_paths,
    RegimeCalibratedParams,
)
from engines.stochastic_engine import run_stochastic_forecast


def test_calibrate_regime_params_no_network_needed(synthetic_returns):
    """
    The whole point of the refactor: calibrate_regime_params must work on a
    pure in-memory returns Series, no fetch_price_history call anywhere. If
    this raises or hangs, something reintroduced a network dependency.
    """
    params = calibrate_regime_params(port_r=synthetic_returns, random_state=42)
    assert isinstance(params, RegimeCalibratedParams)


def test_calibrate_regime_params_structural_invariants(synthetic_returns):
    """
    Real invariants, not golden values: regardless of the exact fitted
    numbers, these must always hold for valid output.
    """
    params = calibrate_regime_params(port_r=synthetic_returns, random_state=42)

    assert params.n_regimes == 3
    assert params.regime_stats.shape == (3, 4)
    assert params.transition_probs.shape == (3, 3)

    # every row of the transition matrix must sum to 1 (it's a probability distribution)
    row_sums = params.transition_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0)

    # current_regime_probs must be a valid probability vector too
    assert params.current_regime_probs.shape == (3,)
    assert np.all(params.current_regime_probs >= 0.0)
    assert params.current_regime_probs.sum() == pytest.approx(1.0)


def test_calibrate_regime_params_matches_golden_reference(synthetic_returns):
    """
    Golden-value regression check: GMM/EM has no closed form, so there's no
    way to independently hand-derive the "correct" regime stats -- these
    values were computed once from the real code and are treated as the
    trusted reference.
    """
    params = calibrate_regime_params(port_r=synthetic_returns, random_state=42)

    expected_means = [0.000266, -0.000110, -0.000730]
    expected_stds = [0.007474, 0.008203, 0.023944]

    assert params.regime_stats["mean"].tolist() == pytest.approx(expected_means, abs=1e-6)
    assert params.regime_stats["std"].tolist() == pytest.approx(expected_stds, abs=1e-6)


def test_simulate_regime_switching_paths_forced_start_stays_put():
    """
    Pure-math unit test, no calibration involved: force every path to start
    in a highly persistent ("sticky") regime and confirm most paths are
    still there after a short horizon -- a statistical bound, not exact
    equality, since it's random.
    """
    regime_stats = pd.DataFrame({
        "mean": [0.0005, -0.002, 0.0],
        "std": [0.01, 0.03, 0.015],
    }, index=[0, 1, 2])

    transition_probs = pd.DataFrame([
        [0.90, 0.00, 0.10],
        [0.00, 0.97, 0.03],   # regime 1 is very sticky
        [0.05, 0.05, 0.90],
    ], index=[0, 1, 2], columns=[0, 1, 2])

    result = simulate_regime_switching_paths(
        start_regime_probs=np.array([0.0, 1.0, 0.0]),  # force start in regime 1
        regime_stats=regime_stats,
        transition_probs=transition_probs,
        s0=100.0, horizon=10, n_paths=5000, random_seed=1,
    )

    assert result["prices"].shape == (5000, 11)
    assert result["variances"].shape == (5000, 11)
    assert np.all(result["prices"][:, 0] == 100.0)

    # After 10 days with 97% daily persistence, expect the large majority
    # of paths to still be in the sticky regime (std=0.03 -> variance=0.0009)
    sticky_variance = 0.03 ** 2
    frac_still_sticky = np.mean(np.isclose(result["variances"][:, -1], sticky_variance))
    assert frac_still_sticky > 0.6


def test_run_stochastic_forecast_regime_output_contract(synthetic_returns):
    params = calibrate_regime_params(port_r=synthetic_returns, random_state=42)

    out = run_stochastic_forecast(
        model="regime", s0=100.0, T=30 / 252, N=30, n=500,
        regime_params=params, random_seed=123,
    )

    assert out["model"] == "regime"
    assert out["paths"].shape == (500, 31)
    assert out["variance_paths"].shape == (500, 31)
    assert "variance" in out
    assert out["params"]["n_regimes"] == 3

    for key in ("terminal", "drawdown", "path_metrics"):
        assert key in out


def test_run_stochastic_forecast_regime_missing_params_raises():
    with pytest.raises(ValueError, match="regime_params is required"):
        run_stochastic_forecast(model="regime", s0=100.0, T=1.0, N=10, n=10)


def test_run_stochastic_forecast_regime_wrong_type_raises():
    with pytest.raises(TypeError, match="RegimeCalibratedParams instance"):
        run_stochastic_forecast(
            model="regime", s0=100.0, T=1.0, N=10, n=10,
            regime_params={"not": "a dataclass"},
        )


def test_run_stochastic_forecast_regime_reproducible_with_fixed_seed(synthetic_returns):
    params = calibrate_regime_params(port_r=synthetic_returns, random_state=42)

    out = run_stochastic_forecast(
        model="regime", s0=100.0, T=30 / 252, N=30, n=500,
        regime_params=params, random_seed=123,
    )

    assert out["terminal"]["mean_terminal_value"] == pytest.approx(99.95652894780441, rel=1e-6)
