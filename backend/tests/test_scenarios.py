import pandas as pd
import pytest

from engines.scenario_engine import (
    apply_price_shock,
    apply_shock_with_linear_rebound,
    apply_regime_shift,
)


@pytest.fixture
def prices_df():
    idx = pd.to_datetime(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    )
    return pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0, 120.0, 130.0],
            "MSFT": [200.0, 200.0, 210.0, 220.0, 215.0],
        },
        index=idx,
    )


# ------------------------
# apply_price_shock tests
# ------------------------

def test_apply_price_shock_applies_on_and_after_date(prices_df):
    shocked = apply_price_shock(prices_df, shock_date="2025-01-06", shock_pct=-0.10)
    factor = 0.90

    # before shock date unchanged
    pd.testing.assert_series_equal(shocked.loc["2025-01-02"], prices_df.loc["2025-01-02"])
    pd.testing.assert_series_equal(shocked.loc["2025-01-03"], prices_df.loc["2025-01-03"])

    # on/after shock date scaled
    for d in ["2025-01-06", "2025-01-07", "2025-01-08"]:
        pd.testing.assert_series_equal(shocked.loc[d], prices_df.loc[d] * factor)


def test_apply_price_shock_does_not_mutate_input(prices_df):
    original = prices_df.copy(deep=True)
    _ = apply_price_shock(prices_df, shock_date="2025-01-06", shock_pct=-0.10)
    pd.testing.assert_frame_equal(prices_df, original)


def test_apply_price_shock_empty_df_returns_copy():
    empty = pd.DataFrame()
    out = apply_price_shock(empty, shock_date="2025-01-06", shock_pct=-0.10)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


# -----------------------------------
# apply_shock_with_linear_rebound tests
# -----------------------------------

def test_linear_rebound_applies_window_and_reaches_baseline(prices_df):
    # shock date between index dates is fine; it will start at the first trading day on/after
    shocked = apply_shock_with_linear_rebound(
        prices_df, shock_date="2025-01-06", shock_pct=-0.10, rebound_days=3
    )

    # start_pos = 2025-01-06
    # rebound_days=3 => end_pos = start_pos + 3 = 2025-01-08 (within data)
    # steps = 2? careful: end_pos = start_pos + rebound_days, inclusive window length rebound_days+1 if not clipped.
    #
    # In your implementation:
    # end_pos = min(start_pos + rebound_days, len(idx)-1)
    # steps = end_pos - start_pos
    # multipliers are length steps+1 from m0 to 1.0 inclusive.
    #
    # Here: start_pos at 01-06 (pos 2), end_pos at pos 5? but we only have 5 rows total (0..4)
    # Actually idx has 5 rows, start_pos=2, start_pos+3=5 -> clipped to 4 (01-08)
    # steps = 4 - 2 = 2 -> multipliers for 01-06, 01-07, 01-08 are [0.9, 0.95, 1.0]
    expected_multipliers = {
        "2025-01-06": 0.90,
        "2025-01-07": 0.95,
        "2025-01-08": 1.00,
    }

    # before shock start unchanged
    pd.testing.assert_series_equal(shocked.loc["2025-01-02"], prices_df.loc["2025-01-02"])
    pd.testing.assert_series_equal(shocked.loc["2025-01-03"], prices_df.loc["2025-01-03"])

    # during rebound window: scaled by expected multipliers
    for d, m in expected_multipliers.items():
        pd.testing.assert_series_equal(shocked.loc[d], prices_df.loc[d] * m)


def test_linear_rebound_rebound_days_must_be_ge_1(prices_df):
    with pytest.raises(ValueError, match="rebound_days must be >= 1"):
        apply_shock_with_linear_rebound(prices_df, shock_date="2025-01-06", shock_pct=-0.10, rebound_days=0)


def test_linear_rebound_shock_date_after_data_no_change(prices_df):
    shocked = apply_shock_with_linear_rebound(
        prices_df, shock_date="2025-12-31", shock_pct=-0.10, rebound_days=5
    )
    pd.testing.assert_frame_equal(shocked, prices_df)


# -------------------------
# apply_regime_shift tests
# -------------------------

def test_regime_shift_no_change_if_shock_date_after_data(prices_df):
    shocked = apply_regime_shift(prices_df, shock_date="2025-12-31", vol_mult=2.0, drift_shift=-0.001)
    pd.testing.assert_frame_equal(shocked, prices_df)


def test_regime_shift_changes_prices_on_or_after_start_date(prices_df):
    # Use a shock date that is exactly on an index date so start_pos is unambiguous
    shocked = apply_regime_shift(prices_df, shock_date="2025-01-06", vol_mult=1.5, drift_shift=-0.001)

    # Anchor day is the day before start_date (2025-01-03) and should remain unchanged
    assert shocked.loc["2025-01-03", "AAPL"] == pytest.approx(prices_df.loc["2025-01-03", "AAPL"])
    assert shocked.loc["2025-01-03", "MSFT"] == pytest.approx(prices_df.loc["2025-01-03", "MSFT"])

    # At least one post-shock date should differ (unless parameters are neutral, which they aren't)
    # Compare 2025-01-06 row
    assert not shocked.loc["2025-01-06"].equals(prices_df.loc["2025-01-06"])

    # Shape and index should match
    assert shocked.shape == prices_df.shape
    assert shocked.index.equals(prices_df.index)
    assert list(shocked.columns) == list(prices_df.columns)
