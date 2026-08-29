"""
Standalone playground/validation script for simulate_regime_switching_paths.

Fits the regime model on REAL historical data (same pipeline as
regime_classifier.py), then runs the simulator two ways so you can compare
them side by side:
  1. STRESS TEST  -- force every path to start in the crisis regime.
  2. NATURAL FORECAST -- start from today's actual nowcasted regime
     probabilities instead.

Tweak HORIZON / N_PATHS / RANDOM_SEED below and re-run to poke at it.
"""
from __future__ import annotations

import numpy as np

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_value_series, portfolio_returns
from engines.ml.models.regime_classifier import (
    build_feature_frames,
    fit_regime_model,
    label_regimes,
    compute_regime_stats,
    compute_transition_probs,
    tickers,
    weights,
    fetch_start,
    fetch_end,
)
from engines.ml.feature_matrix_builder import build_feature_matrix
from engines.ml.rolling_backtest_forecasting_models import simulate_regime_switching_paths

HORIZON = 30
N_PATHS = 2000
RANDOM_SEED = 42


def print_summary(label: str, result: dict) -> None:
    prices = result["prices"]
    terminal_returns = prices[:, -1] / prices[:, 0] - 1.0
    p05, p50, p95 = np.percentile(terminal_returns, [5, 50, 95])

    print(f"=== {label} ===")
    print(f"Mean terminal return:   {terminal_returns.mean():.2%}")
    print(f"Median terminal return: {np.median(terminal_returns):.2%}")
    print(f"p05 / p50 / p95:        {p05:.2%} / {p50:.2%} / {p95:.2%}")
    print(f"Probability of loss:    {(terminal_returns < 0).mean():.2%}")
    print()


def main():
    # --- Fit the regime model on real data (same pipeline as regime_classifier.py) ---
    price_hist = fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    prices = price_hist.prices
    asset_returns = prices_to_returns(prices)
    port_r = portfolio_returns(asset_returns=asset_returns, weights=weights)
    port_p_df = portfolio_value_series(prices, weights).to_frame(name="portfolio")
    port_r_df = port_r.to_frame(name="portfolio")

    feature_frames = build_feature_frames(port_p_df, port_r_df)
    X = build_feature_matrix(feature_frames).dropna()
    gmm, scaler, X_scaled = fit_regime_model(X, n_components=3, covariance_type="full", random_state=RANDOM_SEED)
    labels_df = label_regimes(gmm, X_scaled, X.index)

    regime_stats = compute_regime_stats(port_r, labels_df)
    transition_probs = compute_transition_probs(port_r, labels_df)

    print("=== Fitted regime stats (daily mean/std, annualized versions too) ===")
    print(regime_stats)
    print()
    print("=== Fitted transition matrix ===")
    print(transition_probs)
    print()

    s0 = float(port_p_df["portfolio"].iloc[-1])
    n_regimes = len(regime_stats)

    # --- Mode 1: STRESS TEST -- force start in whichever regime has the
    # HIGHEST volatility (not lowest mean -- regimes trade off mean and vol
    # differently, and vol is what actually matches the "crisis" regime we
    # already validated against real historical dates earlier). Don't
    # hardcode a regime index, since GMM's numbering is arbitrary per fit.
    crisis_regime = int(regime_stats["std"].idxmax())
    forced_start = np.zeros(n_regimes)
    forced_start[crisis_regime] = 1.0

    stress_result = simulate_regime_switching_paths(
        start_regime_probs=forced_start,
        regime_stats=regime_stats,
        transition_probs=transition_probs,
        s0=s0, horizon=HORIZON, n_paths=N_PATHS, random_seed=RANDOM_SEED,
    )
    print_summary(f"STRESS TEST (forced start: regime {crisis_regime})", stress_result)

    # --- Mode 2: NATURAL FORECAST -- nowcast today's actual regime probabilities ---
    latest_feature_row = X.iloc[[-1]]  # double brackets -- keep it a (1, n_features) DataFrame
    X_latest_scaled = scaler.transform(latest_feature_row)
    nowcast_probs = gmm.predict_proba(X_latest_scaled)[0]
    print("Today's nowcasted regime probabilities:", nowcast_probs.round(3))
    print()

    natural_result = simulate_regime_switching_paths(
        start_regime_probs=nowcast_probs,
        regime_stats=regime_stats,
        transition_probs=transition_probs,
        s0=s0, horizon=HORIZON, n_paths=N_PATHS, random_seed=RANDOM_SEED,
    )
    print_summary("NATURAL FORECAST (today's nowcasted regime)", natural_result)


if __name__ == "__main__":
    main()
