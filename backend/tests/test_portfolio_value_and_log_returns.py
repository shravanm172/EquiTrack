# AI Usage Disclosure. This pytest module was written with the assistance of Github Copilot
import numpy as np
import pandas as pd
import pytest

from engines.portfolio_engine import (
    portfolio_value_series,
    portfolio_log_returns_from_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(data: dict, dates=None) -> pd.DataFrame:
    """Build a small price DataFrame with a DatetimeIndex."""
    n = len(next(iter(data.values())))
    if dates is None:
        dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(data, index=dates, dtype="float64")


# ===========================================================================
# portfolio_value_series
# ===========================================================================

class TestPortfolioValueSeries:

    # --- happy-path ----------------------------------------------------------

    def test_single_asset_starts_at_one(self):
        prices = _make_prices({"AAPL": [100.0, 105.0, 110.0]})
        result = portfolio_value_series(prices, {"AAPL": 1.0})
        assert float(result.iloc[0]) == pytest.approx(1.0)

    def test_single_asset_tracks_price_ratio(self):
        prices = _make_prices({"AAPL": [100.0, 110.0, 120.0]})
        result = portfolio_value_series(prices, {"AAPL": 1.0})
        # day 1: 110/100 = 1.1; day 2: 120/100 = 1.2
        assert float(result.iloc[1]) == pytest.approx(1.1)
        assert float(result.iloc[2]) == pytest.approx(1.2)

    def test_equal_weight_two_assets(self):
        # AAPL: flat, MSFT: doubles → portfolio should end at 1.5
        prices = _make_prices({"AAPL": [100.0, 100.0], "MSFT": [100.0, 200.0]})
        result = portfolio_value_series(prices, {"AAPL": 0.5, "MSFT": 0.5})
        assert float(result.iloc[0]) == pytest.approx(1.0)
        assert float(result.iloc[1]) == pytest.approx(1.5)

    def test_output_name_is_portfolio_value(self):
        prices = _make_prices({"AAPL": [100.0, 105.0]})
        result = portfolio_value_series(prices, {"AAPL": 1.0})
        assert result.name == "portfolio_value"

    def test_output_length_matches_input(self):
        prices = _make_prices({"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0]})
        result = portfolio_value_series(prices, {"AAPL": 1.0})
        assert len(result) == len(prices)

    def test_weights_are_normalized(self):
        # weights = {AAPL: 2, MSFT: 2} should behave identically to {0.5, 0.5}
        prices = _make_prices({"AAPL": [100.0, 120.0], "MSFT": [100.0, 80.0]})
        result_raw = portfolio_value_series(prices, {"AAPL": 2.0, "MSFT": 2.0})
        result_norm = portfolio_value_series(prices, {"AAPL": 0.5, "MSFT": 0.5})
        pd.testing.assert_series_equal(result_raw, result_norm)

    def test_tickers_are_case_insensitive(self):
        prices = _make_prices({"AAPL": [100.0, 110.0]})
        lower = portfolio_value_series(prices, {"aapl": 1.0})
        upper = portfolio_value_series(prices, {"AAPL": 1.0})
        pd.testing.assert_series_equal(lower, upper)

    def test_extra_weight_tickers_ignored(self):
        """Weights may contain tickers not in the price DataFrame; they are dropped."""
        prices = _make_prices({"AAPL": [100.0, 110.0]})
        result = portfolio_value_series(prices, {"AAPL": 0.6, "GOOG": 0.4})
        # Only AAPL is in prices; after filtering & re-normalising the portfolio
        # is 100 % AAPL, so it should track AAPL exactly.
        assert float(result.iloc[0]) == pytest.approx(1.0)
        assert float(result.iloc[1]) == pytest.approx(1.1)

    def test_index_matches_price_index(self):
        prices = _make_prices({"AAPL": [100.0, 105.0, 110.0]})
        result = portfolio_value_series(prices, {"AAPL": 1.0})
        pd.testing.assert_index_equal(result.index, prices.index)

    def test_three_assets_weighted(self):
        prices = _make_prices({"A": [100.0, 200.0], "B": [100.0, 100.0], "C": [100.0, 50.0]})
        # 50 % A (doubles), 25 % B (flat), 25 % C (halves)
        # expected terminal = 0.5*2 + 0.25*1 + 0.25*0.5 = 1.0 + 0.25 + 0.125 = 1.375
        result = portfolio_value_series(prices, {"A": 0.5, "B": 0.25, "C": 0.25})
        assert float(result.iloc[1]) == pytest.approx(1.375)

    # --- forward-fill of NaNs ------------------------------------------------

    def test_forward_fill_fills_missing_prices(self):
        """A NaN mid-series should be forward-filled, not dropped.

        dropna(how='all') only removes rows where EVERY column is NaN.
        We need a second column so the NaN row survives that step and
        ffill() can propagate the last valid AAPL price forward.
        """
        prices = _make_prices({"AAPL": [100.0, np.nan, 110.0], "MSFT": [200.0, 210.0, 220.0]})
        # Use only AAPL weight so MSFT is excluded after ticker filtering,
        # but the row must survive dropna(how='all') first.
        # Use equal weights so we can reason about both columns.
        result = portfolio_value_series(prices, {"AAPL": 0.5, "MSFT": 0.5})
        # AAPL ffilled: [100, 100, 110] → normalized [1.0, 1.0, 1.1]
        # MSFT:         [200, 210, 220] → normalized [1.0, 1.05, 1.1]
        # portfolio (equal weight): [1.0, 1.025, 1.1]
        assert len(result) == 3
        assert float(result.iloc[0]) == pytest.approx(1.0)
        assert float(result.iloc[1]) == pytest.approx(0.5 * 1.0 + 0.5 * 1.05)
        assert float(result.iloc[2]) == pytest.approx(0.5 * 1.1 + 0.5 * 1.1)

    # --- edge cases / validation ---------------------------------------------

    def test_empty_prices_raises(self):
        prices = pd.DataFrame(dtype="float64")
        with pytest.raises(ValueError, match="prices must not be empty"):
            portfolio_value_series(prices, {"AAPL": 1.0})

    def test_no_matching_tickers_raises(self):
        prices = _make_prices({"AAPL": [100.0, 105.0]})
        with pytest.raises(ValueError, match="None of the portfolio tickers"):
            portfolio_value_series(prices, {"GOOG": 1.0})

    def test_zero_weight_sum_raises(self):
        prices = _make_prices({"AAPL": [100.0, 105.0]})
        with pytest.raises(ValueError, match="Weights sum to 0"):
            portfolio_value_series(prices, {"AAPL": 0.0})


# ===========================================================================
# portfolio_log_returns_from_value
# ===========================================================================

class TestPortfolioLogReturnsFromValue:

    # --- happy-path ----------------------------------------------------------

    def test_output_length_is_n_minus_one(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        value = pd.Series([1.0, 1.1, 1.05, 1.15, 1.2], index=idx)
        result = portfolio_log_returns_from_value(value)
        assert len(result) == len(value) - 1

    def test_known_log_return_values(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        value = pd.Series([1.0, np.e, np.e**2], index=idx)
        result = portfolio_log_returns_from_value(value)
        # log(e/1) = 1, log(e²/e) = 1
        assert float(result.iloc[0]) == pytest.approx(1.0)
        assert float(result.iloc[1]) == pytest.approx(1.0)

    def test_flat_series_gives_zero_returns(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        value = pd.Series([2.0, 2.0, 2.0, 2.0], index=idx)
        result = portfolio_log_returns_from_value(value)
        assert (result.abs() < 1e-12).all()

    def test_output_name_is_portfolio_log_return(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        value = pd.Series([1.0, 1.1, 1.2], index=idx)
        result = portfolio_log_returns_from_value(value)
        assert result.name == "portfolio_log_return"

    def test_log_return_is_negative_for_declining_value(self):
        idx = pd.bdate_range("2024-01-01", periods=2)
        value = pd.Series([1.0, 0.9], index=idx)
        result = portfolio_log_returns_from_value(value)
        assert float(result.iloc[0]) < 0.0

    def test_log_return_is_positive_for_rising_value(self):
        idx = pd.bdate_range("2024-01-01", periods=2)
        value = pd.Series([1.0, 1.1], index=idx)
        result = portfolio_log_returns_from_value(value)
        assert float(result.iloc[0]) > 0.0

    def test_index_starts_at_second_observation(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        value = pd.Series([1.0, 1.1, 1.2, 1.3], index=idx)
        result = portfolio_log_returns_from_value(value)
        assert result.index[0] == idx[1]

    def test_round_trip_consistency_with_pipeline(self):
        """portfolio_value_series → portfolio_log_returns_from_value should give sane output."""
        prices = _make_prices({
            "AAPL": [100, 102, 101, 105, 108],
            "MSFT": [200, 198, 202, 205, 203],
        })
        value = portfolio_value_series(prices, {"AAPL": 0.5, "MSFT": 0.5})
        log_r = portfolio_log_returns_from_value(value)
        # Reconstructed value should match: exp(cumsum of log returns) * value[0]
        reconstructed = np.exp(log_r.cumsum()) * float(value.iloc[0])
        pd.testing.assert_series_equal(
            reconstructed.rename("portfolio_value"),
            value.iloc[1:].rename("portfolio_value"),
            check_names=True,
            atol=1e-10,
        )

    # --- edge cases / validation ---------------------------------------------

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="portfolio_value must not be empty"):
            portfolio_log_returns_from_value(pd.Series(dtype="float64"))

    def test_single_element_series_returns_empty(self):
        idx = pd.bdate_range("2024-01-01", periods=1)
        value = pd.Series([1.0], index=idx)
        result = portfolio_log_returns_from_value(value)
        assert result.empty
