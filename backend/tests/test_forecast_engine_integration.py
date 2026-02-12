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


@pytest.fixture
def port_returns_mixed():
    """
    Mixed returns so rolling mean differs from full-sample mean.
    """
    idx = pd.to_datetime(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    )
    # full mean = (0 + 0 + 0 + 0.1 + 0.1)/5 = 0.04
    # rolling window 2 mean = (0.1 + 0.1)/2 = 0.1
    return pd.Series([0.0, 0.0, 0.0, 0.10, 0.10], index=idx, name="port_r")


# -----------------------------
# UNIT TEST: _forecast_from_returns (mean)
# -----------------------------
def test_forecast_from_returns_mean_shapes_and_continuity(port_returns_uptrend):
    starting_cash = 100_000.0
    forecast_days = 5

    out = fe._forecast_from_returns(
        port_r=port_returns_uptrend,
        starting_cash=starting_cash,
        forecast_days=forecast_days,
        mode="mean",
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


def test_forecast_from_returns_invalid_days_raises(port_returns_uptrend):
    with pytest.raises(ValueError, match="forecast_days must be > 0"):
        fe._forecast_from_returns(
            port_r=port_returns_uptrend,
            starting_cash=100_000.0,
            forecast_days=0,
            mode="mean",
        )


def test_forecast_from_returns_empty_returns_raises():
    empty = pd.Series([], dtype="float64")
    with pytest.raises(ValueError, match="portfolio returns empty"):
        fe._forecast_from_returns(
            port_r=empty,
            starting_cash=100_000.0,
            forecast_days=5,
            mode="mean",
        )


def test_forecast_from_returns_invalid_mode_raises(port_returns_uptrend):
    with pytest.raises(ValueError, match="forecast\.mode must be"):
        fe._forecast_from_returns(
            port_r=port_returns_uptrend,
            starting_cash=100_000.0,
            forecast_days=5,
            mode="nope",
        )


# -----------------------------
# UNIT TEST: _forecast_from_returns (rolling)
# -----------------------------
def test_forecast_from_returns_rolling_uses_last_window(port_returns_mixed):
    starting_cash = 100_000.0
    forecast_days = 3
    window = 2

    out = fe._forecast_from_returns(
        port_r=port_returns_mixed,
        starting_cash=starting_cash,
        forecast_days=forecast_days,
        mode="rolling",
        window=window,
    )

    assert out["trend"]["mode"] == "rolling"
    assert out["trend"]["window"] == window

    r_hat = float(out["trend"]["mean_daily_return"])
    assert r_hat == pytest.approx(0.10, abs=1e-12)  # last 2 returns are 0.1, 0.1

    # Sanity: forecast grows by 10% per step from last historical value
    hist = out["historical_equity_curve"]
    fc = out["forecast_equity_curve"]
    last_hist_val = float(hist[-1]["value"])
    expected_first_fc = round(last_hist_val * (1.0 + r_hat), 2)
    assert float(fc[0]["value"]) == pytest.approx(expected_first_fc, abs=1e-9)


def test_forecast_from_returns_rolling_window_too_big_raises(port_returns_uptrend):
    with pytest.raises(ValueError, match="Not enough return data for rolling window"):
        fe._forecast_from_returns(
            port_r=port_returns_uptrend,
            starting_cash=100_000.0,
            forecast_days=3,
            mode="rolling",
            window=999,
        )


def test_forecast_from_returns_rolling_window_nonpositive_raises(port_returns_uptrend):
    with pytest.raises(ValueError, match="forecast.window must be > 0"):
        fe._forecast_from_returns(
            port_r=port_returns_uptrend,
            starting_cash=100_000.0,
            forecast_days=3,
            mode="rolling",
            window=0,
        )


# -----------------------------
# UNIT TEST: _forecast_from_returns (rolling)
# -----------------------------
def test_forecast_from_returns_ewma_alpha_computes_expected_drift():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    port_r = pd.Series([0.0, 0.0, 0.10], index=idx, name="port_r")

    out = fe._forecast_from_returns(
        port_r=port_r,
        starting_cash=100_000.0,
        forecast_days=3,
        mode="ewma",
        alpha=0.5,
    )

    assert out["trend"]["mode"] == "ewma"
    assert out["trend"]["alpha"] == pytest.approx(0.5, abs=1e-12)
    assert out["trend"]["lambda"] == pytest.approx(0.5, abs=1e-12)
    assert float(out["trend"]["mean_daily_return"]) == pytest.approx(0.05, abs=1e-12)


def test_forecast_from_returns_ewma_defaults_lambda(port_returns_uptrend):
    out = fe._forecast_from_returns(
        port_r=port_returns_uptrend,   # now this is the Series
        starting_cash=100_000.0,
        forecast_days=3,
        mode="ewma",
    )
    assert out["trend"]["mode"] == "ewma"
    assert out["trend"]["lambda"] == pytest.approx(0.94, abs=1e-12)
    assert out["trend"]["alpha"] == pytest.approx(0.06, abs=1e-12)


def test_forecast_from_returns_ewma_invalid_alpha_raises(port_returns_uptrend):
    with pytest.raises(ValueError, match="forecast.alpha must be in \\(0, 1\\)"):
        fe._forecast_from_returns(
            port_r=port_returns_uptrend,
            starting_cash=100_000.0,
            forecast_days=3,
            mode="ewma",
            alpha=1.0,
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
                "date_range": {"start": "2025-01-02", "end": "2025-01-08"},
                "weights": {"AAPL": 0.5, "MSFT": 0.5},
            },
            "portfolio_returns": port_r,
            "last_equity_date": port_r.index[-1],
            "last_equity_value": 0.0,  # unused by current forecaster (it rebuilds curve)
        }
    )


def test_forecast_endpoint_mean_returns_expected_shape(client, port_returns_uptrend):
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

    assert out["trend"]["mode"] == "mean"
    assert "mean_daily_return" in out["trend"]

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


def test_forecast_endpoint_rolling_echoes_window(client, port_returns_mixed):
    analysis_id = _seed_analysis_in_store(port_returns_mixed)

    forecast_days = 5
    window = 2
    resp = client.post(
        "/api/forecast",
        json={
            "analysis_id": analysis_id,
            "forecast": {"days": forecast_days, "mode": "rolling", "window": window},
        },
    )
    assert resp.status_code == 200

    out = resp.get_json()
    assert out["trend"]["mode"] == "rolling"
    assert out["trend"]["window"] == window

    assert out["inputs"]["forecast"]["mode"] == "rolling"
    assert out["inputs"]["forecast"]["window"] == window
    assert out["inputs"]["forecast"]["days"] == forecast_days


def test_forecast_endpoint_missing_analysis_id_returns_400(client):
    resp = client.post("/api/forecast", json={"forecast": {"days": 5}})
    assert resp.status_code == 400
    out = resp.get_json()
    assert "error" in out


def test_forecast_endpoint_expired_or_unknown_analysis_id_returns_400(client):
    resp = client.post(
        "/api/forecast",
        json={"analysis_id": "doesnotexist", "forecast": {"days": 5}},
    )
    assert resp.status_code == 400
    out = resp.get_json()
    assert "error" in out


def test_forecast_endpoint_invalid_mode_returns_400(client, port_returns_uptrend):
    analysis_id = _seed_analysis_in_store(port_returns_uptrend)

    resp = client.post(
        "/api/forecast",
        json={"analysis_id": analysis_id, "forecast": {"days": 5, "mode": "badmode"}},
    )
    assert resp.status_code == 400
    out = resp.get_json()
    assert "error" in out


def test_forecast_endpoint_ewma_echoes_alpha(client, port_returns_mixed):
    analysis_id = _seed_analysis_in_store(port_returns_mixed)

    resp = client.post(
        "/api/forecast",
        json={
            "analysis_id": analysis_id,
            "forecast": {"days": 5, "mode": "ewma", "alpha": 0.5},
        },
    )
    assert resp.status_code == 200
    out = resp.get_json()

    assert out["trend"]["mode"] == "ewma"
    assert out["trend"]["alpha"] == pytest.approx(0.5, abs=1e-12)
    assert out["inputs"]["forecast"]["mode"] == "ewma"
    assert out["inputs"]["forecast"]["alpha"] == pytest.approx(0.5, abs=1e-12)



def test_forecast_endpoint_ewma_defaults_lambda(client, port_returns_mixed):
    analysis_id = _seed_analysis_in_store(port_returns_mixed)

    resp = client.post(
        "/api/forecast",
        json={
            "analysis_id": analysis_id,
            "forecast": {"days": 5, "mode": "ewma"},
        },
    )
    assert resp.status_code == 200
    out = resp.get_json()

    assert out["trend"]["mode"] == "ewma"
    assert out["trend"]["lambda"] == pytest.approx(0.94, abs=1e-12)
    assert out["inputs"]["forecast"]["mode"] == "ewma"
    assert out["inputs"]["forecast"]["lambda"] == pytest.approx(0.94, abs=1e-12)