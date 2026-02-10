import pandas as pd
import pytest

from engines.portfolio_engine import prices_to_returns


def test_prices_to_returns_basic_math():
    # 3 dates, 2 tickers
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 99.0],
            "MSFT": [200.0, 220.0, 242.0],
        },
        index=idx,
    )

    r = prices_to_returns(prices)

    # first row dropped because pct_change produces NaN there
    assert list(r.index.strftime("%Y-%m-%d")) == ["2025-01-03", "2025-01-06"]

    # AAPL: 110/100 - 1 = 0.10
    assert r.loc["2025-01-03", "AAPL"] == pytest.approx(0.10)

    # MSFT: 220/200 - 1 = 0.10
    assert r.loc["2025-01-03", "MSFT"] == pytest.approx(0.10)

    # AAPL: 99/110 - 1 = -0.10
    assert r.loc["2025-01-06", "AAPL"] == pytest.approx(-0.10)

    # MSFT: 242/220 - 1 = 0.10
    assert r.loc["2025-01-06", "MSFT"] == pytest.approx(0.10)


def test_prices_to_returns_drops_all_nan_rows_but_keeps_partial():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    prices = pd.DataFrame(
        {
            # AAPL missing on 01-03
            "AAPL": [100.0, None, 110.0],
            "MSFT": [200.0, 220.0, 242.0],
        },
        index=idx,
    )

    r = prices_to_returns(prices)

    # 2025-01-03: AAPL return will be NaN, MSFT return valid -> row kept
    assert "2025-01-03" in set(r.index.strftime("%Y-%m-%d"))
    assert pd.isna(r.loc["2025-01-03", "AAPL"])
    assert r.loc["2025-01-03", "MSFT"] == pytest.approx(0.10)

    # 2025-01-06: AAPL return still NaN because prior price was NaN; MSFT valid -> row kept
    assert "2025-01-06" in set(r.index.strftime("%Y-%m-%d"))
    assert pd.isna(r.loc["2025-01-06", "AAPL"])
    assert r.loc["2025-01-06", "MSFT"] == pytest.approx(242.0 / 220.0 - 1.0)


def test_prices_to_returns_empty_input_returns_empty_df():
    prices = pd.DataFrame()
    r = prices_to_returns(prices)
    assert isinstance(r, pd.DataFrame)
    assert r.empty
