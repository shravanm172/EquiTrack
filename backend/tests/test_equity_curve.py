# AI Disclosure: This file includes content generated with GPT-5.2.
import numpy as np
import pandas as pd
import pytest

from engines.analytics_engine import equity_curve


def test_equity_curve_compounds_correctly():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    portfolio_returns = pd.Series(
        [0.10, -0.05, 0.20],
        index=idx,
        name="portfolio_return",
    )

    curve = equity_curve(portfolio_returns, starting_cash=100.0)

    # Day-by-day:
    # Day1: 100 * 1.10 = 110
    # Day2: 110 * 0.95 = 104.5
    # Day3: 104.5 * 1.20 = 125.4
    assert curve.loc["2025-01-02"] == pytest.approx(110.0)
    assert curve.loc["2025-01-03"] == pytest.approx(104.5)
    assert curve.loc["2025-01-06"] == pytest.approx(125.4)


def test_equity_curve_scales_with_starting_cash():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    portfolio_returns = pd.Series([0.10, 0.10], index=idx)

    curve_100 = equity_curve(portfolio_returns, starting_cash=100.0)
    curve_1000 = equity_curve(portfolio_returns, starting_cash=1000.0)

    ratio = (curve_1000 / curve_100).to_numpy()
    np.testing.assert_allclose(ratio, 10.0, rtol=1e-12, atol=1e-12)


def test_equity_curve_empty_returns():
    portfolio_returns = pd.Series(dtype="float64")
    curve = equity_curve(portfolio_returns, starting_cash=100.0)

    assert isinstance(curve, pd.Series)
    assert curve.empty
    assert curve.name == "equity"


def test_equity_curve_preserves_index_and_name():
    idx = pd.to_datetime(["2025-01-02"])
    portfolio_returns = pd.Series([0.05], index=idx)

    curve = equity_curve(portfolio_returns, starting_cash=200.0)

    assert curve.index.equals(idx)
    assert curve.name == "equity"
