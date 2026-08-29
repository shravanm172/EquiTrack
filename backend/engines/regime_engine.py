from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from engines.ml.feature_matrix_builder import build_feature_matrix
from engines.ml.models.regime_classifier import (
    build_feature_frames,
    fit_regime_model,
    label_regimes,
    compute_regime_stats,
    compute_transition_probs,
)

N_REGIMES_DEFAULT = 3


@dataclass
class RegimeCalibratedParams:
    portfolio_name: str
    start: str
    end: str
    n_regimes: int

    s0: float                          # synthetic reconstructed portfolio value on the last date (see calibrate_regime_params)
    regime_stats: pd.DataFrame        # mean/std/annualized_return/annualized_vol per regime
    transition_probs: pd.DataFrame    # n_regimes x n_regimes empirical Markov transition matrix
    current_regime_probs: np.ndarray  # nowcasted regime probabilities as of the last date in port_r


def nowcast_regime(gmm, scaler, latest_feature_row: pd.DataFrame) -> np.ndarray:
    """
    Identify the current regime -- the starting point for forecasting.

    Parameters:
        - GaussianMixture (already fitted)
        - StandardScaler (already fitted)
        - features from the most recent date, as a single-row DataFrame

    Returns:
        numpy ndarray with the soft probabilities that the portfolio is
        currently in each regime state
    """
    X_scaled = scaler.transform(latest_feature_row)
    curr_regime_probs = gmm.predict_proba(X_scaled)
    return curr_regime_probs


def calibrate_regime_params(
    port_r: pd.Series,
    n_regimes: int = N_REGIMES_DEFAULT,
    portfolio_name: str = "portfolio",
    random_state: int = 42,
) -> RegimeCalibratedParams:
    """
    Fit the regime GMM on an already-computed portfolio return series
    (reusing regime_classifier.py's validated pipeline) and bundle
    everything a regime-switching simulation needs into one
    calibrated-params object. Matches calibrate_garch_mle's contract --
    no network fetch, just the returns series the caller already has.

    The drawdown feature needs a price-like series, not just returns.
    Rather than fetching real prices, reconstruct a synthetic one via
    cumulative compounding from an arbitrary reference value of 1.0 --
    the exact same technique analytics_engine.equity_curve already uses
    for every model's s0 (V_t = start * cumprod(1+r_t)). Drawdown is a
    scale-invariant ratio (distance from a running peak, as a %), so this
    gives identical drawdown values to using real prices.

    Parameters:
        - port_r: portfolio daily returns series
        - number of regime states to fit GMM
        - name of portfolio
        - random seed
    Returns:
        - RegimeCalibratedParams object with aggregate regime stats, transition probability matrix, \
          and nowcasted regime probabilities for the last date in port_r
    """
    synthetic_prices = (1.0 + port_r).cumprod()
    port_p_df = synthetic_prices.to_frame(name="portfolio")
    port_r_df = port_r.to_frame(name="portfolio")

    feature_frames = build_feature_frames(port_prices=port_p_df, port_returns=port_r_df)
    X = build_feature_matrix(features=feature_frames).dropna()

    gmm, scaler, X_scaled = fit_regime_model(
        X=X, n_components=n_regimes, covariance_type="full", random_state=random_state,
    )
    labels_df = label_regimes(gmm=gmm, X_scaled=X_scaled, index=X.index)

    regime_stats = compute_regime_stats(port_returns=port_r, labels_df=labels_df)
    transition_probs = compute_transition_probs(port_returns=port_r, labels_df=labels_df)

    latest_feature_row = X.iloc[[-1]]  # double brackets keeps it a (1, n_features) DataFrame
    current_regime_probs = nowcast_regime(gmm=gmm, scaler=scaler, latest_feature_row=latest_feature_row)[0]

    s0 = float(port_p_df["portfolio"].iloc[-1])

    return RegimeCalibratedParams(
        portfolio_name=portfolio_name,
        start=port_r.index[0].strftime("%Y-%m-%d"),
        end=port_r.index[-1].strftime("%Y-%m-%d"),
        n_regimes=n_regimes,
        s0=s0,
        regime_stats=regime_stats,
        transition_probs=transition_probs,
        current_regime_probs=current_regime_probs,
    )


def simulate_regime_switching_paths(
    start_regime_probs: np.ndarray,
    regime_stats: pd.DataFrame,
    transition_probs: pd.DataFrame,
    s0: float,
    horizon: int,
    n_paths: int,
    random_seed: int,
) -> dict:
    """
    Every day, every path does two independent random draws:
      1. Draw today's return from the current regime's own fitted (mean, std).
      2. Draw tomorrow's regime from that regime's own row of
         transition_probs.
    
    Parameters:
        - start_regime_probs: shape (n_regimes,), 1D e.g. today's nowcasted
          probabilities for a natural forecast, or a one-hot [0,0,1] to force
          a specific starting regime (crisis) for a stress test.
        - regime_stats: from compute_regime_stats -- indexed 0..n_regimes-1, with
          daily "mean"/"std" columns 
        - transition_probs: from compute_transition_probs -- (n_regimes, n_regimes),
          rows/columns both indexed 0..n_regimes-1.
        - s0: initial portfolio value
        - horizon: future window -- (days)
        - n_paths: number of paths to simulate

    Returns: 
        - {"prices": (n_paths, horizon+1), "variances": (n_paths, horizon+1)}
    """
    rng = np.random.default_rng(random_seed)
    n_regimes = len(regime_stats)

    means = regime_stats["mean"].to_numpy()   # (n_regimes,) daily mean per regime
    stds = regime_stats["std"].to_numpy()     # (n_regimes,) daily std per regime
    trans_cumprobs = transition_probs.to_numpy().cumsum(axis=1)  # (n_regimes, n_regimes)

    prices = np.empty((n_paths, horizon + 1), dtype=float)
    variances = np.empty((n_paths, horizon + 1), dtype=float)
    prices[:, 0] = s0

    # (start_regime_probs) broadcast against every path at once.
    start_regime_probs = np.asarray(start_regime_probs, dtype=float).reshape(-1)
    start_cum = np.cumsum(start_regime_probs)
    u0 = rng.random(n_paths)
    current_regime = (u0.reshape(-1, 1) > start_cum.reshape(1, -1)).sum(axis=1)
    current_regime = np.clip(current_regime, 0, n_regimes - 1)  # float-rounding safety at the p=1.0 edge

    variances[:, 0] = stds[current_regime] ** 2

    for t in range(horizon):
        # 1. Today's return, using each path's current regime's own mean/std
        r_t = means[current_regime] + stds[current_regime] * rng.standard_normal(n_paths)
        prices[:, t + 1] = prices[:, t] * (1.0 + r_t)

        # 2. Tomorrow's regime, using each path's current regime's transition row
        row_for_each_path = trans_cumprobs[current_regime]       
        u = rng.random(n_paths).reshape(-1, 1)
        current_regime = (u > row_for_each_path).sum(axis=1)
        current_regime = np.clip(current_regime, 0, n_regimes - 1)

        variances[:, t + 1] = stds[current_regime] ** 2

    return {"prices": prices, "variances": variances}
