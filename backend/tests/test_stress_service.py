"""
Integration tests for the stress-testing service layer
(services/stress_service.py): run_deterministic_regime_stress_forecast,
preview_calibrated_regime_states, run_calibrated_regime_stress_test.

Verification, not validation (same philosophy as
test_run_stochastic_forecast_gbm.py / test_run_stochastic_forecast_regime.py):
those already cover the underlying GBM / regime-switching Monte Carlo math
at the run_stochastic_forecast level. These tests check that
stress_service.py wires everything correctly on top of that -- right data
flows through (cached returns, as-of-date truncation, the shared response
shape both transforms are supposed to share), right errors raised, right
shapes returned -- using the deterministic synthetic_returns fixture.

No HTTP-route tests here yet -- app.py hasn't been wired to the new
functions/routes yet, that's the next step.
"""
import pandas as pd
import pytest

from services.store_singleton import analysis_store
from services.stress_service import (
    run_deterministic_regime_stress_forecast,
    preview_calibrated_regime_states,
    run_calibrated_regime_stress_test,
)


def _seed_analysis(port_r: pd.Series, starting_cash: float = 100_000.0) -> str:
    """Seed an 'analyze'-kind cache entry directly, matching what /api/analyze
    would have produced, so stress_service can be tested without a prior
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
# run_deterministic_regime_stress_forecast
# -----------------------------

def test_deterministic_returns_expected_shape(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    out = run_deterministic_regime_stress_forecast({
        "analysis_id": analysis_id, "days": 20, "simulations": 300, "random_seed": 1,
    })

    assert out["inputs"]["shock"]["type"] == "deterministic_regime"
    assert out["inputs"]["days"] == 20
    assert out["inputs"]["simulations"] == 300

    assert len(out["historical_equity_curve"]) > 0
    for key in ("p10", "p25", "p50", "p75", "p90"):
        assert len(out["forecast_paths"][key]) == 20

    for key in ("terminal", "drawdown"):
        assert key in out
    # GBM has no time-varying variance path -- no "variance" key, unlike
    # calibrated_regime.
    assert "variance" not in out


def test_deterministic_default_drift_shift_and_vol_mult(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    out = run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "simulations": 200})
    assert out["inputs"]["shock"]["drift_shift"] == pytest.approx(-0.0005)
    assert out["inputs"]["shock"]["vol_mult"] == pytest.approx(1.5)


def test_deterministic_default_as_of_date_is_last_cached_date(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    out = run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "simulations": 200})

    last_cached_date = synthetic_returns.index[-1].strftime("%Y-%m-%d")
    assert out["inputs"]["as_of_date"]["applied"] == last_cached_date
    assert out["actual_realized_curve"] == []


def test_deterministic_as_of_date_in_past_produces_nonempty_actual_realized_curve(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    as_of = synthetic_returns.index[500].strftime("%Y-%m-%d")

    out = run_deterministic_regime_stress_forecast({
        "analysis_id": analysis_id, "as_of_date": as_of, "days": 10, "simulations": 200,
    })

    assert out["inputs"]["as_of_date"]["applied"] == as_of
    assert len(out["actual_realized_curve"]) > 0
    assert out["historical_equity_curve"][-1]["date"] == as_of

    last_cached_date = synthetic_returns.index[-1].strftime("%Y-%m-%d")
    assert out["actual_realized_curve"][-1]["date"] == last_cached_date


def test_deterministic_forecast_dates_start_after_as_of_date(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    as_of = synthetic_returns.index[500].strftime("%Y-%m-%d")

    out = run_deterministic_regime_stress_forecast({
        "analysis_id": analysis_id, "as_of_date": as_of, "days": 10, "simulations": 200,
    })

    as_of_ts = pd.to_datetime(as_of)
    first_forecast_date = pd.to_datetime(out["forecast_paths"]["p50"][0]["date"])
    assert first_forecast_date > as_of_ts


def test_deterministic_regression_guard_s0_is_real_dollar_scale(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    out = run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "simulations": 200})
    assert out["historical_equity_curve"][-1]["value"] > 1000


def test_deterministic_vol_mult_widens_outcome_spread(synthetic_returns):
    """Behavioral sanity check, not a golden value: a bigger vol_mult should
    produce a meaningfully wider bull/bear spread. Catches a wiring bug
    where vol_mult silently has no effect."""
    analysis_id = _seed_analysis(synthetic_returns)

    calm = run_deterministic_regime_stress_forecast({
        "analysis_id": analysis_id, "vol_mult": 1.0, "days": 30, "simulations": 1000, "random_seed": 7,
    })
    stormy = run_deterministic_regime_stress_forecast({
        "analysis_id": analysis_id, "vol_mult": 3.0, "days": 30, "simulations": 1000, "random_seed": 7,
    })

    calm_spread = calm["terminal"]["bull_case"] - calm["terminal"]["bear_case"]
    stormy_spread = stormy["terminal"]["bull_case"] - stormy["terminal"]["bear_case"]
    assert stormy_spread > calm_spread


def test_deterministic_drift_shift_lowers_terminal_mean(synthetic_returns):
    """Behavioral sanity check: a more negative drift_shift should lower the
    mean terminal value. Catches a wiring bug where drift_shift silently has
    no effect."""
    analysis_id = _seed_analysis(synthetic_returns)

    mild = run_deterministic_regime_stress_forecast({
        "analysis_id": analysis_id, "drift_shift": 0.0, "days": 60, "simulations": 2000, "random_seed": 3,
    })
    severe = run_deterministic_regime_stress_forecast({
        "analysis_id": analysis_id, "drift_shift": -0.01, "days": 60, "simulations": 2000, "random_seed": 3,
    })

    assert severe["terminal"]["mean_terminal_value"] < mild["terminal"]["mean_terminal_value"]


def test_deterministic_missing_analysis_id_raises():
    with pytest.raises(ValueError, match="analysis_id is required"):
        run_deterministic_regime_stress_forecast({})


def test_deterministic_unknown_analysis_id_raises():
    with pytest.raises(ValueError, match="not found or expired"):
        run_deterministic_regime_stress_forecast({"analysis_id": "doesnotexist"})


def test_deterministic_invalid_source_raises(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="source must be"):
        run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "source": "nonsense"})


@pytest.mark.parametrize("days", [0, -5])
def test_deterministic_invalid_days_raises(synthetic_returns, days):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="days must be > 0"):
        run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "days": days})


@pytest.mark.parametrize("simulations", [0, -100])
def test_deterministic_invalid_simulations_raises(synthetic_returns, simulations):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="simulations must be > 0"):
        run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "simulations": simulations})


def test_deterministic_invalid_vol_mult_raises(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="vol_mult must be > 0"):
        run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "vol_mult": 0})


def test_deterministic_as_of_date_after_cached_window_raises(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="after the last date"):
        run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "as_of_date": "2099-01-01"})


def test_deterministic_malformed_as_of_date_raises(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="as_of_date must be YYYY-MM-DD"):
        run_deterministic_regime_stress_forecast({"analysis_id": analysis_id, "as_of_date": "not-a-date"})


# -----------------------------
# preview_calibrated_regime_states
# -----------------------------

def test_preview_returns_expected_shape(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    out = preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": 3})

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
        assert sum(row) == pytest.approx(1.0, abs=1e-6)

    assert len(out["current_regime_probs"]) == 3
    assert sum(out["current_regime_probs"]) == pytest.approx(1.0, abs=1e-6)

    last_cached_date = synthetic_returns.index[-1].strftime("%Y-%m-%d")
    assert out["as_of_date"]["applied"] == last_cached_date


def test_preview_default_n_regimes_is_three(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    out = preview_calibrated_regime_states({"analysis_id": analysis_id})
    assert out["n_regimes"] == 3


def test_preview_as_of_date_truncates_fit(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    as_of = synthetic_returns.index[500].strftime("%Y-%m-%d")

    out = preview_calibrated_regime_states({"analysis_id": analysis_id, "as_of_date": as_of, "n_regimes": 3})
    assert out["as_of_date"]["applied"] == as_of


def test_preview_missing_analysis_id_raises():
    with pytest.raises(ValueError, match="analysis_id is required"):
        preview_calibrated_regime_states({})


def test_preview_unknown_analysis_id_raises():
    with pytest.raises(ValueError, match="not found or expired"):
        preview_calibrated_regime_states({"analysis_id": "doesnotexist"})


def test_preview_invalid_source_raises(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="source must be"):
        preview_calibrated_regime_states({"analysis_id": analysis_id, "source": "nonsense"})


@pytest.mark.parametrize("n_regimes", [1, 9])
def test_preview_n_regimes_out_of_range_raises(synthetic_returns, n_regimes):
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="n_regimes must be between"):
        preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": n_regimes})


# -----------------------------
# run_calibrated_regime_stress_test
# -----------------------------

def test_run_full_flow(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": 3})

    out = run_calibrated_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": 0,
        "days": 20, "simulations": 300, "random_seed": 1,
    })

    assert out["inputs"]["shock"]["type"] == "calibrated_regime"
    assert out["inputs"]["shock"]["selected_state"] == 0
    assert out["inputs"]["shock"]["n_regimes"] == 3
    assert out["inputs"]["days"] == 20

    assert len(out["historical_equity_curve"]) > 0
    assert out["historical_equity_curve"][-1]["value"] > 1000  # real-dollar-s0 regression guard

    for key in ("p10", "p25", "p50", "p75", "p90"):
        assert len(out["forecast_paths"][key]) == 20

    for key in ("terminal", "drawdown", "variance"):
        assert key in out


def test_run_actual_realized_curve_empty_when_as_of_is_last_cached_date(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": 3})

    out = run_calibrated_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": 0, "days": 10, "simulations": 200,
    })
    assert out["actual_realized_curve"] == []


def test_run_actual_realized_curve_nonempty_when_as_of_in_past(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    as_of = synthetic_returns.index[500].strftime("%Y-%m-%d")
    preview = preview_calibrated_regime_states({"analysis_id": analysis_id, "as_of_date": as_of, "n_regimes": 3})

    out = run_calibrated_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": 0, "days": 10, "simulations": 200,
    })

    assert len(out["actual_realized_curve"]) > 0
    last_cached_date = synthetic_returns.index[-1].strftime("%Y-%m-%d")
    assert out["actual_realized_curve"][-1]["date"] == last_cached_date


def test_run_forecast_dates_start_after_as_of_date(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": 3})

    out = run_calibrated_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": 0, "days": 10, "simulations": 200,
    })

    last_hist_date = pd.to_datetime(out["historical_equity_curve"][-1]["date"])
    first_forecast_date = pd.to_datetime(out["forecast_paths"]["p50"][0]["date"])
    assert first_forecast_date > last_hist_date


def test_run_selected_state_changes_outcome(synthetic_returns):
    """
    Behavioral sanity check, not a golden value: stress-testing the highest-
    volatility state should produce a meaningfully wider outcome spread than
    the lowest-volatility state. Catches a wiring bug where selected_state
    silently has no effect.
    """
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": 3})

    stats_by_state = {s["state"]: s for s in preview["regime_stats"]}
    calmest = min(stats_by_state, key=lambda s: stats_by_state[s]["annualized_volatility"])
    stormiest = max(stats_by_state, key=lambda s: stats_by_state[s]["annualized_volatility"])
    assert calmest != stormiest

    calm_out = run_calibrated_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": calmest,
        "days": 30, "simulations": 1000, "random_seed": 5,
    })
    stormy_out = run_calibrated_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": stormiest,
        "days": 30, "simulations": 1000, "random_seed": 5,
    })

    calm_spread = calm_out["terminal"]["bull_case"] - calm_out["terminal"]["bear_case"]
    stormy_spread = stormy_out["terminal"]["bull_case"] - stormy_out["terminal"]["bear_case"]
    assert stormy_spread > calm_spread


def test_run_missing_regime_id_raises():
    with pytest.raises(ValueError, match="regime_id is required"):
        run_calibrated_regime_stress_test({"selected_state": 0})


def test_run_unknown_regime_id_raises():
    with pytest.raises(ValueError, match="not found or expired"):
        run_calibrated_regime_stress_test({"regime_id": "doesnotexist", "selected_state": 0})


def test_run_missing_selected_state_raises(synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": 3})
    with pytest.raises(ValueError, match="selected_state is required"):
        run_calibrated_regime_stress_test({"regime_id": preview["regime_id"]})


@pytest.mark.parametrize("selected_state", [-1, 3, 99])
def test_run_selected_state_out_of_range_raises(synthetic_returns, selected_state):
    analysis_id = _seed_analysis(synthetic_returns)
    preview = preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": 3})
    with pytest.raises(ValueError, match="selected_state must be between"):
        run_calibrated_regime_stress_test({"regime_id": preview["regime_id"], "selected_state": selected_state})


def test_run_rejects_non_preview_id(synthetic_returns):
    """A regime_id must come from preview_calibrated_regime_states -- an
    ordinary analyze-kind analysis_id shouldn't be accepted."""
    analysis_id = _seed_analysis(synthetic_returns)
    with pytest.raises(ValueError, match="does not refer to a calibrated-regime preview"):
        run_calibrated_regime_stress_test({"regime_id": analysis_id, "selected_state": 0})


# -----------------------------
# shared response-shape symmetry
# -----------------------------

def test_both_transforms_share_the_same_response_shape(synthetic_returns):
    """The whole point of _build_stress_response: both transforms should
    return the same top-level keys (deterministic just omits "variance",
    since GBM has no time-varying variance path), and the same "inputs"
    sub-keys."""
    analysis_id = _seed_analysis(synthetic_returns)

    deterministic_out = run_deterministic_regime_stress_forecast({
        "analysis_id": analysis_id, "days": 10, "simulations": 200,
    })

    preview = preview_calibrated_regime_states({"analysis_id": analysis_id, "n_regimes": 3})
    calibrated_out = run_calibrated_regime_stress_test({
        "regime_id": preview["regime_id"], "selected_state": 0, "days": 10, "simulations": 200,
    })

    assert set(deterministic_out.keys()) == set(calibrated_out.keys()) - {"variance"}
    assert set(deterministic_out["inputs"].keys()) == set(calibrated_out["inputs"].keys())


# -----------------------------
# HTTP routes
# -----------------------------

def test_stress_deterministic_regime_endpoint_returns_expected_shape(client, synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    resp = client.post("/api/stress/deterministic_regime", json={
        "analysis_id": analysis_id, "days": 15, "simulations": 200,
    })
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["inputs"]["shock"]["type"] == "deterministic_regime"
    assert len(data["forecast_paths"]["p50"]) == 15
    assert "variance" not in data


def test_stress_deterministic_regime_endpoint_respects_shock_params(client, synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    resp = client.post("/api/stress/deterministic_regime", json={
        "analysis_id": analysis_id, "days": 10, "simulations": 200,
        "drift_shift": -0.01, "vol_mult": 2.5,
    })
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["inputs"]["shock"]["drift_shift"] == pytest.approx(-0.01)
    assert data["inputs"]["shock"]["vol_mult"] == pytest.approx(2.5)


def test_stress_deterministic_regime_endpoint_missing_analysis_id_returns_400(client):
    resp = client.post("/api/stress/deterministic_regime", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_stress_deterministic_regime_endpoint_unknown_analysis_id_returns_400(client):
    resp = client.post("/api/stress/deterministic_regime", json={"analysis_id": "doesnotexist"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_stress_calibrated_regime_preview_endpoint_returns_expected_shape(client, synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    resp = client.post("/api/stress/calibrated_regime/preview", json={
        "analysis_id": analysis_id, "n_regimes": 3,
    })
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["n_regimes"] == 3
    assert len(data["regime_stats"]) == 3
    assert "regime_id" in data


def test_stress_calibrated_regime_preview_endpoint_missing_analysis_id_returns_400(client):
    resp = client.post("/api/stress/calibrated_regime/preview", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_stress_calibrated_regime_stress_endpoint_full_flow(client, synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)

    preview_resp = client.post("/api/stress/calibrated_regime/preview", json={
        "analysis_id": analysis_id, "n_regimes": 3,
    })
    preview = preview_resp.get_json()

    stress_resp = client.post("/api/stress/calibrated_regime/stress", json={
        "regime_id": preview["regime_id"], "selected_state": 0,
        "days": 15, "simulations": 200,
    })
    assert stress_resp.status_code == 200

    data = stress_resp.get_json()
    assert data["inputs"]["shock"]["type"] == "calibrated_regime"
    assert data["inputs"]["shock"]["selected_state"] == 0
    assert len(data["forecast_paths"]["p50"]) == 15
    assert "variance" in data


def test_stress_calibrated_regime_stress_endpoint_unknown_regime_id_returns_400(client):
    resp = client.post("/api/stress/calibrated_regime/stress", json={
        "regime_id": "doesnotexist", "selected_state": 0,
    })
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_stress_calibrated_regime_stress_endpoint_missing_selected_state_returns_400(client, synthetic_returns):
    analysis_id = _seed_analysis(synthetic_returns)
    preview_resp = client.post("/api/stress/calibrated_regime/preview", json={
        "analysis_id": analysis_id, "n_regimes": 3,
    })
    preview = preview_resp.get_json()

    resp = client.post("/api/stress/calibrated_regime/stress", json={"regime_id": preview["regime_id"]})
    assert resp.status_code == 400
    assert "error" in resp.get_json()
