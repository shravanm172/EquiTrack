"""
Tests for the regime-selecting stress test: preview_regime_states +
run_regime_stress_test (services/regime_service.py), plus the
/api/regime/preview + /api/regime/stress HTTP routes.

Verification, not validation (same philosophy as
test_run_stochastic_forecast_regime.py): these check that the service/route
wiring is correct -- right data flows through, right errors raised, right
shapes returned -- using the deterministic synthetic_returns fixture. The
underlying regime math itself is already covered there.
"""
import pandas as pd
import pytest

from services.store_singleton import analysis_store
from services.regime_service import preview_regime_states, run_regime_stress_test


def _seed_analysis(port_r: pd.Series, starting_cash: float = 100_000.0) -> str:
    """Seed an 'analyze'-kind cache entry directly, matching what /api/analyze
    would have produced, so regime_service can be tested without a prior
    real analysis call."""
    return analysis_store.put({
        "kind": "analyze",
        "inputs": {
            "starting_cash": float(starting_cash),
            "weights": {"AAPL": 1.0},
            "date_range": {"start": "2018-01-01", "end": "2020-12-31"},
        },
        "portfolio_returns": port_r,
    })


# -----------------------------
# preview_regime_states
# -----------------------------

def test_preview_regime_states_returns_expected_shape(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    out = preview_regime_states({"analysis_id": analysis_id, "source": "baseline", "n_regimes": 3})

    assert "regime_id" in out
    assert out["n_regimes"] == 3
    assert len(out["regime_stats"]) == 3
    for row in out["regime_stats"]:
        assert set(row.keys()) == {
            "state", "mean_daily_return", "daily_volatility",
            "annualized_return", "annualized_volatility",
        }

    assert len(out["transition_probs"]) == 3
    for row in out["transition_probs"]:
        assert len(row) == 3
        assert row == pytest.approx(row)  # sanity: all finite floats
        assert sum(row) == pytest.approx(1.0, abs=1e-6)

    assert len(out["current_regime_probs"]) == 3
    assert sum(out["current_regime_probs"]) == pytest.approx(1.0, abs=1e-6)


def test_preview_regime_states_default_n_regimes_is_three(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    out = preview_regime_states({"analysis_id": analysis_id})
    assert out["n_regimes"] == 3


def test_preview_regime_states_missing_analysis_id_raises():
    with pytest.raises(ValueError, match="analysis_id is required"):
        preview_regime_states({})


def test_preview_regime_states_unknown_analysis_id_raises():
    with pytest.raises(ValueError, match="not found or expired"):
        preview_regime_states({"analysis_id": "doesnotexist"})


def test_preview_regime_states_invalid_source_raises(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="source must be"):
        preview_regime_states({"analysis_id": analysis_id, "source": "nonsense"})


@pytest.mark.parametrize("n_regimes", [1, 9])
def test_preview_regime_states_n_regimes_out_of_range_raises(synthetic_returns, n_regimes):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="n_regimes must be between"):
        preview_regime_states({"analysis_id": analysis_id, "n_regimes": n_regimes})


# -----------------------------
# run_regime_stress_test
# -----------------------------

def test_run_regime_stress_test_full_flow(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_regime_states({"analysis_id": analysis_id, "n_regimes": 3})

    out = run_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": 0,
        "days": 20, "simulations": 300, "random_seed": 1,
    })

    assert out["inputs"]["selected_state"] == 0
    assert out["inputs"]["n_regimes"] == 3
    assert out["inputs"]["days"] == 20

    assert len(out["historical_equity_curve"]) > 0
    # regression guard: s0 must be the real dollar-scale value, not
    # calibrate_regime_params' internal synthetic ~1.0-scale reconstruction
    assert out["historical_equity_curve"][-1]["value"] > 1000

    for key in ("p10", "p25", "p50", "p75", "p90"):
        assert len(out["forecast_paths"][key]) == 20

    for key in ("terminal", "drawdown", "variance"):
        assert key in out


def test_run_regime_stress_test_forecast_dates_start_after_history(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_regime_states({"analysis_id": analysis_id, "n_regimes": 3})

    out = run_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": 0,
        "days": 10, "simulations": 200, "random_seed": 1,
    })

    last_hist_date = pd.to_datetime(out["historical_equity_curve"][-1]["date"])
    first_forecast_date = pd.to_datetime(out["forecast_paths"]["p50"][0]["date"])
    assert first_forecast_date > last_hist_date


def test_run_regime_stress_test_selected_state_changes_outcome(synthetic_returns):
    """
    Behavioral sanity check, not a golden value: stress-testing the highest-
    volatility state should produce a meaningfully wider outcome spread than
    the lowest-volatility state. Catches a wiring bug where selected_state
    silently has no effect.
    """
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_regime_states({"analysis_id": analysis_id, "n_regimes": 3})

    stats_by_state = {s["state"]: s for s in preview["regime_stats"]}
    calmest = min(stats_by_state, key=lambda s: stats_by_state[s]["annualized_volatility"])
    stormiest = max(stats_by_state, key=lambda s: stats_by_state[s]["annualized_volatility"])
    assert calmest != stormiest

    calm_out = run_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": calmest,
        "days": 30, "simulations": 1000, "random_seed": 5,
    })
    stormy_out = run_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": stormiest,
        "days": 30, "simulations": 1000, "random_seed": 5,
    })

    calm_spread = calm_out["terminal"]["bull_case"] - calm_out["terminal"]["bear_case"]
    stormy_spread = stormy_out["terminal"]["bull_case"] - stormy_out["terminal"]["bear_case"]
    assert stormy_spread > calm_spread


def test_run_regime_stress_test_missing_regime_id_raises():
    with pytest.raises(ValueError, match="regime_id is required"):
        run_regime_stress_test({"selected_state": 0})


def test_run_regime_stress_test_unknown_regime_id_raises():
    with pytest.raises(ValueError, match="not found or expired"):
        run_regime_stress_test({"regime_id": "doesnotexist", "selected_state": 0})


def test_run_regime_stress_test_missing_selected_state_raises(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_regime_states({"analysis_id": analysis_id, "n_regimes": 3})
    with pytest.raises(ValueError, match="selected_state is required"):
        run_regime_stress_test({"regime_id": preview["regime_id"]})


@pytest.mark.parametrize("selected_state", [-1, 3, 99])
def test_run_regime_stress_test_selected_state_out_of_range_raises(synthetic_returns, selected_state):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_regime_states({"analysis_id": analysis_id, "n_regimes": 3})
    with pytest.raises(ValueError, match="selected_state must be between"):
        run_regime_stress_test({"regime_id": preview["regime_id"], "selected_state": selected_state})


def test_run_regime_stress_test_rejects_non_preview_id(synthetic_returns):
    """A regime_id must come from preview_regime_states -- an ordinary
    analyze-kind analysis_id shouldn't be accepted."""
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="does not refer to a regime preview"):
        run_regime_stress_test({"regime_id": analysis_id, "selected_state": 0})


# -----------------------------
# HTTP routes
# -----------------------------

def test_regime_preview_endpoint_returns_expected_shape(client, synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    resp = client.post("/api/regime/preview", json={"analysis_id": analysis_id, "n_regimes": 3})
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["n_regimes"] == 3
    assert len(data["regime_stats"]) == 3
    assert "regime_id" in data


def test_regime_stress_endpoint_full_flow(client, synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    preview_resp = client.post("/api/regime/preview", json={"analysis_id": analysis_id, "n_regimes": 3})
    preview = preview_resp.get_json()

    stress_resp = client.post("/api/regime/stress", json={
        "regime_id": preview["regime_id"], "selected_state": 0,
        "days": 15, "simulations": 200,
    })
    assert stress_resp.status_code == 200

    data = stress_resp.get_json()
    assert data["inputs"]["selected_state"] == 0
    assert len(data["forecast_paths"]["p50"]) == 15


def test_regime_preview_endpoint_missing_analysis_id_returns_400(client):
    resp = client.post("/api/regime/preview", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_regime_stress_endpoint_unknown_regime_id_returns_400(client):
    resp = client.post("/api/regime/stress", json={"regime_id": "doesnotexist", "selected_state": 0})
    assert resp.status_code == 400
    assert "error" in resp.get_json()
