import pandas as pd
import pytest

from services.analysis_service import analyze_portfolio


class DummyPH:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices


@pytest.fixture
def prices_df():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    return pd.DataFrame(
        {"AAPL": [100.0, 110.0, 121.0], "MSFT": [200.0, 200.0, 200.0]},
        index=idx,
    )


@pytest.fixture
def mock_fetch_price_history(monkeypatch, prices_df):
    import services.analysis_service as svc

    def fake_fetch(tickers, start, end):
        return DummyPH(prices_df)

    monkeypatch.setattr(svc, "fetch_price_history", fake_fetch)


def test_analyze_portfolio_weights_mode_builds_portfolio_and_runs(mock_fetch_price_history):
    payload = {
        "portfolio": {
            "starting_cash": 100.0,
            "holdings": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5},
            ],
        },
        "date_range": {"start": "2025-01-02", "end": "2025-01-06"},
    }

    out = analyze_portfolio(payload)

    assert out["inputs"]["mode"] == "weights"
    assert out["inputs"]["starting_cash"] == pytest.approx(100.0)
    assert out["inputs"]["weights"]["AAPL"] == pytest.approx(0.5)
    assert out["inputs"]["weights"]["MSFT"] == pytest.approx(0.5)

    # weights mode should not include holdings_breakdown
    assert "holdings_breakdown" not in out

    # equity curve should exist
    assert len(out["equity_curve"]) == 2


def test_analyze_portfolio_shares_mode_converts_to_weights_and_defaults_cash(
    mock_fetch_price_history,
):
    payload = {
        "portfolio": {
            # omit starting_cash to test default
            "holdings": [
                {"ticker": "AAPL", "shares": 10},
                {"ticker": "MSFT", "shares": 5},
            ],
        },
        "date_range": {"start": "2025-01-02", "end": "2025-01-06"},
    }

    out = analyze_portfolio(payload)

    assert out["inputs"]["mode"] == "shares"
    assert "holdings_breakdown" in out

    hb = out["holdings_breakdown"]
    assert hb["as_of"] == "2025-01-02"

    # Total value = 10*100 + 5*200 = 1000 + 1000 = 2000
    assert hb["total_value"] == pytest.approx(2000.0, abs=0.01)

    # starting_cash should default to total_value
    assert out["inputs"]["starting_cash"] == pytest.approx(2000.0, abs=0.01)

    # Converted weights should be ~50/50
    assert out["inputs"]["weights"]["AAPL"] == pytest.approx(0.5, abs=1e-6)
    assert out["inputs"]["weights"]["MSFT"] == pytest.approx(0.5, abs=1e-6)

    assert len(out["equity_curve"]) == 2
