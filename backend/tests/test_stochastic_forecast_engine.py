# tests/test_stochastic_forecast_engine.py

import numpy as np
import pytest

from engines.stochastic_engine import (
    simulate_gbm_path,
    simulate_many_paths,
    summarize_terminal_metrics,
    summarize_drawdown_metrics,
    summarize_path_metrics,
)


def test_simulate_gbm_path_returns_length_n_plus_1():
    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    T = 1.0
    N = 252

    path = simulate_gbm_path(s0, mu, sigma, T, N)

    assert len(path) == N + 1
    assert path[0] == s0


def test_simulate_gbm_path_all_values_positive():
    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    T = 1.0
    N = 252

    path = simulate_gbm_path(s0, mu, sigma, T, N)

    assert all(x > 0 for x in path)


def test_simulate_gbm_path_zero_volatility_is_deterministic():
    s0 = 100.0
    mu = 0.08
    sigma = 0.0
    T = 1.0
    N = 4

    path = simulate_gbm_path(s0, mu, sigma, T, N)

    dt = T / N
    expected = [s0]
    cur = s0
    for _ in range(N):
        cur = cur * np.exp(mu * dt)
        expected.append(cur)

    assert np.allclose(path, expected, rtol=1e-12, atol=1e-12)


def test_simulate_many_paths_returns_array_with_correct_shape():
    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    T = 1.0
    N = 252
    n = 100

    paths = simulate_many_paths(s0, mu, sigma, T, N, n)

    assert isinstance(paths, np.ndarray)
    assert paths.shape == (n, N + 1)


def test_simulate_many_paths_first_column_equals_s0():
    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    T = 1.0
    N = 10
    n = 20

    paths = simulate_many_paths(s0, mu, sigma, T, N, n)

    assert np.allclose(paths[:, 0], s0)


def test_simulate_many_paths_all_values_positive():
    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    T = 1.0
    N = 50
    n = 25

    paths = simulate_many_paths(s0, mu, sigma, T, N, n)

    assert np.all(paths > 0)


def test_summarize_terminal_metrics_on_known_paths():
    paths = np.array(
        [
            [100.0, 110.0, 120.0],
            [100.0, 90.0, 80.0],
            [100.0, 105.0, 100.0],
        ]
    )

    out = summarize_terminal_metrics(paths)

    terminal_values = np.array([120.0, 80.0, 100.0])

    assert np.allclose(out["terminal_values"], terminal_values)
    assert out["mean_terminal_value"] == pytest.approx(np.mean(terminal_values))
    assert out["median_terminal_value"] == pytest.approx(np.median(terminal_values))
    assert out["bear_case"] == pytest.approx(np.percentile(terminal_values, 10))
    assert out["bull_case"] == pytest.approx(np.percentile(terminal_values, 90))
    assert out["probability_of_loss"] == pytest.approx(1 / 3)


def test_summarize_drawdown_metrics_on_known_paths():
    paths = np.array(
        [
            [100.0, 120.0, 90.0],   # max drawdown = -25%
            [100.0, 110.0, 105.0],  # max drawdown = -5/110
            [100.0, 95.0, 80.0],    # max drawdown = -20%
        ]
    )

    out = summarize_drawdown_metrics(paths)

    expected_max_drawdowns = np.array([
        -0.25,
        (105.0 - 110.0) / 110.0,
        -0.20,
    ])

    assert np.allclose(out["max_drawdowns"], expected_max_drawdowns)
    assert out["median_max_drawdown"] == pytest.approx(np.median(expected_max_drawdowns))
    assert out["prob_drawdown_gt_20"] == pytest.approx(2 / 3)


def test_summarize_path_metrics_on_known_paths():
    paths = np.array(
        [
            [100.0, 110.0, 120.0],
            [100.0, 90.0, 80.0],
            [100.0, 105.0, 100.0],
        ]
    )

    out = summarize_path_metrics(paths)

    assert np.allclose(out["p50_path"], np.median(paths, axis=0))
    assert np.allclose(out["p10_path"], np.percentile(paths, 10, axis=0))
    assert np.allclose(out["p90_path"], np.percentile(paths, 90, axis=0))


def test_terminal_mean_is_reasonably_close_to_theoretical_expectation():
    """
    Statistical sanity check:
    E[S(T)] = S0 * exp(mu * T) for GBM under the engine's parameterization.

    This is not an exact test, so we use a tolerance.
    """
    np.random.seed(42)

    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    T = 1.0
    N = 252
    n = 20000

    paths = simulate_many_paths(s0, mu, sigma, T, N, n)
    terminal_values = paths[:, -1]

    observed_mean = np.mean(terminal_values)
    theoretical_mean = s0 * np.exp(mu * T)

    # 3% relative tolerance is usually safe for this sample size
    assert observed_mean == pytest.approx(theoretical_mean, rel=0.03)


def test_zero_volatility_many_paths_are_identical():
    s0 = 100.0
    mu = 0.08
    sigma = 0.0
    T = 1.0
    N = 12
    n = 10

    paths = simulate_many_paths(s0, mu, sigma, T, N, n)

    first_path = paths[0]
    for i in range(1, n):
        assert np.allclose(paths[i], first_path, rtol=1e-12, atol=1e-12)