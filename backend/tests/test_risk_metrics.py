import math
import pandas as pd
import pytest

from engines.analytics_engine import (
    annualized_volatility,
    max_drawdown,
    annualized_return,
    sharpe_ratio,
)


def test_annualized_volatility_matches_sample_std_scaling():
    # Two daily returns -> sample std is well-defined (ddof=1)
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    r = pd.Series([0.01, -0.01], index=idx)

    # Sample std for [0.01, -0.01]:
    # mean = 0
    # deviations = [0.01, -0.01]
    # sumsq = 0.0001 + 0.0001 = 0.0002
    # variance (ddof=1) = 0.0002 / 1 = 0.0002
    # std = sqrt(0.0002)
    expected_daily_std = math.sqrt(0.0002)
    expected = expected_daily_std * math.sqrt(252)

    out = annualized_volatility(r, periods_per_year=252)
    assert out == pytest.approx(expected)


def test_max_drawdown_simple_curve():
    # Curve: 100 -> 120 -> 90 -> 110
    # Running max: 100, 120, 120, 120
    # Drawdowns: 0, 0, 90/120 - 1 = -0.25, 110/120 - 1 = -0.08333...
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    curve = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)

    out = max_drawdown(curve)
    assert out == pytest.approx(-0.25)


def test_annualized_return_known_time_span():
    # Construct a curve that doubles over exactly 365.25 days -> 1 year
    # Annualized return should be (2)^(1/1) - 1 = 1.0
    idx = pd.to_datetime(["2025-01-01", "2026-01-01"])
    # days between 2025-01-01 and 2026-01-01 is 365 days (not 365.25)
    # So instead of aiming for exactly 1 year, compute expected with your formula.

    curve = pd.Series([100.0, 200.0], index=idx)

    days = (idx[-1] - idx[0]).days
    years = days / 365.25
    expected = (200.0 / 100.0) ** (1.0 / years) - 1.0

    out = annualized_return(curve)
    assert out == pytest.approx(expected)


def test_sharpe_ratio_zero_rf_known_series():
    # Use a simple 3-point return series where mean/std are easy.
    # r = [0.01, 0.02, 0.00]
    # mean = 0.01
    # sample std (ddof=1):
    # deviations = [0, 0.01, -0.01]
    # sumsq = 0 + 0.0001 + 0.0001 = 0.0002
    # variance = 0.0002 / 2 = 0.0001
    # std = 0.01
    # sharpe_daily = 0.01 / 0.01 = 1
    # sharpe_annual = 1 * sqrt(252)
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    r = pd.Series([0.01, 0.02, 0.00], index=idx)

    expected = math.sqrt(252)

    out = sharpe_ratio(r, risk_free_rate_annual=0.0, periods_per_year=252)
    assert out == pytest.approx(expected)
