# AI Disclosure: This file includes content generated with GPT-5.2.
import math
import pandas as pd
import pytest


from services.analysis_service import _analyze_from_prices


def test__analyze_from_prices_end_to_end_with_mock_prices():
    # Prices are deterministic and chosen so returns are simple:
    # AAPL: 100 -> 110 -> 121  (10%, 10%)
    # MSFT: 200 -> 200 -> 200  (0%, 0%)
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 121.0], "MSFT": [200.0, 200.0, 200.0]},
        index=idx,
    )

    weights = {"AAPL": 0.5, "MSFT": 0.5}
    starting_cash = 100.0

    out = _analyze_from_prices(prices, weights, starting_cash)

    # ---- Structure checks (high-signal, non-duplicative) ----
    assert set(out.keys()) == {"equity_curve", "metrics"}
    assert isinstance(out["equity_curve"], list)
    assert isinstance(out["metrics"], dict)

    metrics = out["metrics"]
    for k in ["annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio"]:
        assert k in metrics

    # ---- Equity curve checks ----
    curve = out["equity_curve"]

    # Because pct_change drops the first day, we expect 2 curve points (for 01-03 and 01-06)
    assert len(curve) == 2
    assert curve[0]["date"] == "2025-01-03"
    assert curve[1]["date"] == "2025-01-06"

    # Portfolio daily return each day:
    # day1 = 0.5*(0.10) + 0.5*(0.00) = 0.05
    # day2 = 0.5*(0.10) + 0.5*(0.00) = 0.05
    # Equity:
    # 01-03: 100 * 1.05 = 105
    # 01-06: 105 * 1.05 = 110.25
    assert curve[0]["value"] == pytest.approx(105.00, abs=0.01)
    assert curve[1]["value"] == pytest.approx(110.25, abs=0.01)

    # ---- Metric sanity checks (don’t duplicate unit tests) ----
    assert math.isfinite(metrics["annualized_return"])
    assert math.isfinite(metrics["annualized_volatility"])
    assert math.isfinite(metrics["max_drawdown"])
    assert math.isfinite(metrics["sharpe_ratio"])

    # With monotonic increasing equity, max drawdown should be 0
    assert metrics["max_drawdown"] == pytest.approx(0.0)

    # Volatility should be >= 0
    assert metrics["annualized_volatility"] >= 0.0
