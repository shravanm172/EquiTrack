# AI Disclosure: This file includes content generated with GPT-5.2.
import pandas as pd
import pytest

# Adjust this import to your actual module path
# e.g. from services.stress_service import analyze_with_shock
from services.stress_service import analyze_with_shock


class DummyPH:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices


@pytest.fixture
def prices_df():
    # Deterministic price panel with a clear shock date inside it
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    return pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0, 120.0],
            "MSFT": [200.0, 200.0, 210.0, 220.0],
        },
        index=idx,
    )


@pytest.fixture
def mock_fetch_price_history(monkeypatch, prices_df):
    # Patch WHERE IT'S USED: in the module that defines analyze_with_shock
    import services.stress_service as svc

    def fake_fetch(tickers, start, end):
        return DummyPH(prices_df)

    monkeypatch.setattr(svc, "fetch_price_history", fake_fetch)


def _base_payload(shock_type="permanent", pct=-0.10):
    return {
        "portfolio": {
            "starting_cash": 100000,
            "holdings": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5},
            ],
        },
        "date_range": {"start": "2025-01-02", "end": "2025-01-07"},
        "shock": {
            "type": shock_type,
            "date": "2025-01-06",
            "pct": pct,
            # include optional fields (safe defaults)
            "rebound_days": 2,
            "vol_mult": 1.5,
            "drift_shift": -0.0005,
        },
    }


def test_analyze_with_shock_returns_expected_shape_and_delta_keys(mock_fetch_price_history):
    out = analyze_with_shock(_base_payload("permanent", pct=-0.10))

    # Top-level structure
    assert "inputs" in out
    assert "baseline" in out
    assert "scenario" in out
    assert "delta" in out
    assert "metrics" in out["delta"]

    # Baseline/scenario structure
    for section in ["baseline", "scenario"]:
        assert "equity_curve" in out[section]
        assert "metrics" in out[section]

    # Delta metrics keys match baseline metrics keys
    assert set(out["delta"]["metrics"].keys()) == set(out["baseline"]["metrics"].keys())


def test_analyze_with_shock_zero_pct_produces_zero_deltas(mock_fetch_price_history):
    out = analyze_with_shock(_base_payload("permanent", pct=0.0))

    # When shock_pct=0, apply_price_shock should return identical prices => identical metrics
    for k, v in out["delta"]["metrics"].items():
        assert v == pytest.approx(0.0, abs=1e-10)


def test_analyze_with_shock_permanent_negative_shock_changes_metrics(mock_fetch_price_history):
    out = analyze_with_shock(_base_payload("permanent", pct=-0.10))

    # We don't assert exact values (that would duplicate lower-level tests),
    # but we do want to see some meaningful change.
    delta = out["delta"]["metrics"]

    # A permanent -10% price level shock on/after 2025-01-06 will distort returns around that date,
    # so at least one metric should change.
    assert any(abs(v) > 1e-12 for v in delta.values())


def test_analyze_with_shock_linear_rebound_runs(mock_fetch_price_history):
    out = analyze_with_shock(_base_payload("linear_rebound", pct=-0.10))
    assert out["inputs"]["shock"]["pct"] == pytest.approx(-0.10)
    assert "metrics" in out["delta"]


def test_analyze_with_shock_regime_shift_runs(mock_fetch_price_history):
    payload = _base_payload("regime_shift", pct=0.0)  # pct unused in regime_shift, but allowed
    out = analyze_with_shock(payload)
    assert "metrics" in out["delta"]


def test_analyze_with_shock_unknown_type_raises(mock_fetch_price_history):
    with pytest.raises(ValueError, match="Unknown shock.type"):
        analyze_with_shock(_base_payload("made_up_type", pct=-0.10))
