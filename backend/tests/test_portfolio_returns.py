import pandas as pd
import pytest

from engines.portfolio_engine import portfolio_returns


def test_portfolio_returns_empty_asset_returns():
    asset_returns = pd.DataFrame()
    r = portfolio_returns(asset_returns, {"AAPL": 0.5})
    assert isinstance(r, pd.Series)
    assert r.empty
    assert r.name == "portfolio_return"


def test_portfolio_returns_raises_if_no_matching_tickers():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    asset_returns = pd.DataFrame({"AAPL": [0.01, 0.02]}, index=idx)

    with pytest.raises(ValueError, match="None of the portfolio tickers exist"):
        portfolio_returns(asset_returns, {"MSFT": 1.0})


def test_portfolio_returns_raises_if_weights_sum_to_zero():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    asset_returns = pd.DataFrame({"AAPL": [0.01, 0.02], "MSFT": [0.03, 0.04]}, index=idx)

    with pytest.raises(ValueError, match="Weights sum to 0"):
        portfolio_returns(asset_returns, {"AAPL": 0.0, "MSFT": 0.0})


def test_portfolio_returns_normalizes_weights_and_matches_expected_values():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    asset_returns = pd.DataFrame(
        {
            "AAPL": [0.10, -0.10],
            "MSFT": [0.00, 0.20],
        },
        index=idx,
    )

    # Weights sum to 2.0, so function should normalize to 0.5 / 0.5
    r = portfolio_returns(asset_returns, {"AAPL": 1.0, "MSFT": 1.0})

    # Expected:
    # day1 = 0.5*0.10 + 0.5*0.00 = 0.05
    # day2 = 0.5*(-0.10) + 0.5*0.20 = 0.05
    assert r.loc["2025-01-02"] == pytest.approx(0.05)
    assert r.loc["2025-01-03"] == pytest.approx(0.05)
    assert r.name == "portfolio_return"


def test_portfolio_returns_fills_missing_returns_with_zero():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    asset_returns = pd.DataFrame(
        {
            "AAPL": [0.10, None],
            "MSFT": [None, 0.20],
        },
        index=idx,
    )

    r = portfolio_returns(asset_returns, {"AAPL": 0.5, "MSFT": 0.5})

    # Missing treated as 0.0:
    # day1 = 0.5*0.10 + 0.5*0.00 = 0.05
    # day2 = 0.5*0.00 + 0.5*0.20 = 0.10
    assert r.loc["2025-01-02"] == pytest.approx(0.05)
    assert r.loc["2025-01-03"] == pytest.approx(0.10)


def test_portfolio_returns_handles_mixed_case_tickers_in_weights():
    idx = pd.to_datetime(["2025-01-02"])
    asset_returns = pd.DataFrame({"AAPL": [0.10], "MSFT": [0.20]}, index=idx)

    r = portfolio_returns(asset_returns, {"aapl": 0.25, "MsFt": 0.75})

    # Expected: 0.25*0.10 + 0.75*0.20 = 0.175
    assert r.iloc[0] == pytest.approx(0.175)
