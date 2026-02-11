import pandas as pd
import pytest

from engines import forecast_engine as fe
from services.store_singleton import analysis_store


@pytest.fixture
def port_returns_uptrend():
    """
    Deterministic portfolio returns series: +10% each step.
    Index dates should be business days / trading days.
    """
    idx = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"])
    return pd.Series([0.10, 0.10, 0.10], index=idx, name="port_r")


# -----------------------------
# UNIT TEST: _forecast_from_returns_mean
# -----------------------------
def test_forecast_from_returns_mean_shapes_and_continuity(port_returns_uptrend):
    starting_cash = 100_000.0
    forecast_days = 5

    out = fe._forecast_from_returns_mean(
        port_r=port_returns_uptrend,
        starting_cash=starting_cash,
        forecast_days=forecast_days,
    )

    # Required keys
    assert "trend" in out
    assert "historical_equity_curve" in out
    assert "forecast_equity_curve" in out
    assert "equity_curve" in out

    # Trend info
    assert out["trend"]["mode"] == "mean"
    r_hat = out["trend"]["mean_daily_return"]
    assert isinstance(r_hat, float)
    assert r_hat == pytest.approx(0.10, abs=1e-12)

    hist = out["historical_equity_curve"]
    fc = out["forecast_equity_curve"]
    combined = out["equity_curve"]

    # Length checks
    assert len(hist) > 0
    assert len(fc) == forecast_days
    assert len(combined) == len(hist) + len(fc)

    # Date boundary checks
    last_hist_date = pd.to_datetime(hist[-1]["date"])
    first_fc_date = pd.to_datetime(fc[0]["date"])
    assert first_fc_date > last_hist_date

    # Continuity check (rounded to 2 decimals like service)
    last_hist_val = float(hist[-1]["value"])
    expected_first_fc = round(last_hist_val * (1.0 + float(r_hat)), 2)
    assert float(fc[0]["value"]) == pytest.approx(expected_first_fc, abs=1e-9)


def test_forecast_from_returns_mean_invalid_days_raises(port_returns_uptrend):
    with pytest.raises(ValueError, match="forecast_days must be > 0"):
        fe._forecast_from_returns_mean(
            port_r=port_returns_uptrend,
            starting_cash=100_000.0,
            forecast_days=0,
        )


def test_forecast_from_returns_mean_empty_returns_raises():
    empty = pd.Series([], dtype="float64")
    with pytest.raises(ValueError, match="portfolio returns empty"):
        fe._forecast_from_returns_mean(
            port_r=empty,
            starting_cash=100_000.0,
            forecast_days=5,
        )


# -----------------------------
# INTEGRATION TEST: /api/forecast endpoint
# -----------------------------
@pytest.fixture
def client():
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _seed_analysis_in_store(port_r: pd.Series, starting_cash: float = 100_000.0) -> str:
    """
    Seed the in-memory cache directly so forecast endpoint can run without /api/analyze.
    """
    return analysis_store.put(
        {
            "kind": "analyze",
            "inputs": {
                "mode": "weights",
                "starting_cash": float(starting_cash),
                "date_range": {"start": "2025-01-02", "end": "2025-01-07"},
                "weights": {"AAPL": 0.5, "MSFT": 0.5},
            },
            "portfolio_returns": port_r,
            "last_equity_date": port_r.index[-1],
            "last_equity_value": 0.0,  # unused by current forecaster (it rebuilds curve)
        }
    )


def test_forecast_endpoint_returns_expected_shape(client, port_returns_uptrend):
    analysis_id = _seed_analysis_in_store(port_returns_uptrend)

    forecast_days = 7
    resp = client.post(
        "/api/forecast",
        json={
            "analysis_id": analysis_id,
            "forecast": {"days": forecast_days, "mode": "mean"},
        },
    )
    assert resp.status_code == 200

    out = resp.get_json()
    assert "inputs" in out
    assert "trend" in out

    # Curves
    assert "historical_equity_curve" in out
    assert "forecast_equity_curve" in out
    assert "equity_curve" in out

    assert len(out["forecast_equity_curve"]) == forecast_days
    assert len(out["equity_curve"]) == len(out["historical_equity_curve"]) + forecast_days

    # Inputs echo
    assert out["inputs"]["analysis_id"] == analysis_id
    assert out["inputs"]["forecast"]["days"] == forecast_days
    assert out["inputs"]["forecast"]["mode"] == "mean"
    assert out["inputs"]["source"] == "baseline"


def test_forecast_endpoint_missing_analysis_id_returns_400(client):
    resp = client.post("/api/forecast", json={"forecast": {"days": 5}})
    assert resp.status_code == 400
    out = resp.get_json()
    assert "error" in out


def test_forecast_endpoint_expired_or_unknown_analysis_id_returns_400(client):
    resp = client.post("/api/forecast", json={"analysis_id": "doesnotexist", "forecast": {"days": 5}})
    assert resp.status_code == 400
    out = resp.get_json()
    assert "error" in out
