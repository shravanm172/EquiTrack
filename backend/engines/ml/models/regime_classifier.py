"""
Portfolio-level market regime detector using a Gaussian Mixture Model.

Pipeline (built step by step):
  1. build_regime_feature_frames -- compute the 3 regime-describing features
     (vol level, drawdown, vol short/long ratio) on the portfolio's own
     aggregated price/return series.
  2. build_feature_matrix (reused as-is from feature_matrix_builder.py) --
     stack those into one (date x feature) matrix.
  3. fit_regime_model -- scale + fit the GaussianMixture (the EM step).
  4. label_regimes -- get hard labels + soft probabilities per date.
  5. characterize_regimes -- calibrate each regime's own return/vol stats +
     empirical transition probabilities -- what regime-conditional stress
     testing will eventually pull from.
  6. main -- orchestrate everything, including the known-crisis overlay
     sanity check (March 2020, 2022 selloff, etc.) before trusting the
     regimes for anything downstream.

We're starting with #1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_value_series, portfolio_returns
from engines.features.volatility_features import rolling_volatility, vol_short_long_ratio
from engines.features.drawdown_features import drawdown
from engines.ml.feature_matrix_builder import build_feature_matrix

# Config
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
weights = {"AAPL": 0.2, "MSFT": 0.2, "AMZN": 0.2, "GOOGL": 0.2, "NVDA": 0.2}
fetch_start = "2017-03-01"
fetch_end = "2026-03-23"
N_REGIMES = 3


def build_feature_frames(
    port_prices: pd.DataFrame,
    port_returns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Portfolio prices and returns are passed as parameters
    Computes the features that the regime classifier model uses

    Returns:
        {feature_name: single-column DataFrame}, indexed by date 
    """
    rolling_vol_20_df = rolling_volatility(port_returns)
    drawdown_df = drawdown(port_prices)
    vol_short_long_ratio_df = vol_short_long_ratio(port_returns)

    feature_set = {
        "rolling_vol_20": rolling_vol_20_df,
        "drawdown": drawdown_df,
        "vol_short_long_ratio": vol_short_long_ratio_df,
    }

    return feature_set


def fit_regime_model(
    X: pd.DataFrame,
    n_components: int = N_REGIMES,
    covariance_type: str = "full",
    random_state: int = 42,
) -> tuple[GaussianMixture, StandardScaler, np.ndarray]:
    """
    Applies z-score standardization to features X
    Fits the GaussianMixture via Expectation-Maximization.

    Returns:
        gmm: the fitted GaussianMixture model
        scaler: the fitted StandardScaler 
        X_scaled: the scaled feature matrix used for fitting the model
    """

    # cleaning
    X.dropna(inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)      # note that scaling converts the pandas DataFrame into a Numpy array

    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    gmm.fit(X=X_scaled)

    return gmm, scaler, X_scaled


def label_regimes(gmm: GaussianMixture, X_scaled: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """
    Get hard labels + soft probabilities for every date.

    Note that the index passed must already be aligned with the scaled feature matrix

    Returns:
        DataFrame indexed by `index` with columns:
            "label"                                 -- int, most likely regime (0..n_components-1)
            "prob_regime_0", "prob_regime_1", ...   -- soft probabilities
    """

    # same math but .predict just gives you the argmax
    labels = gmm.predict(X_scaled) 
    probs = gmm.predict_proba(X_scaled)

    # build the return df, indexed by date, with soft probabilities that the date belongs to each regime, and the most likely regime
    columns = []
    for i in range(probs.shape[1]):
        columns.append(f"prob_regime_{i}")
    classified_data = pd.DataFrame(probs, index=index, columns=columns)
    classified_data['label'] = labels

    return classified_data


def compute_regime_stats(port_returns:pd.Series, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the annualized return and annualized volatility for each regime state 
    This allows us to better understand how the model has classified these different regimes

    Parameters:
        - the portfolio returns series
        - the dataframe with hard labels and soft probabilities for each regime -> from label_regimes

    Returns:
        pandas DataFrame with the aggregated stats (ann_return, ann_volatility) for each regime state
    """

    df = port_returns.to_frame("return").join(labels_df["label"], how="inner") # merge them on an inner join to align indices
    stats = df.groupby(['label'])['return'].agg(['mean', 'std'])
    stats['annualized_return'] = stats['mean'] * 252
    stats['annualized_vol'] = stats['std'] * np.sqrt(252)

    return stats

def compute_transition_probs(port_returns: pd.Series, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the probability of transitioning into each regime state, from each regime state
    Will be the basis of the stress testing Monte Carlo sim
    
    Parameters:
            - the portfolio returns series
            - the dataframe with hard labels and soft probabilities for each regime -> from label_regimes
    
    Returns
        pandas DataFrame with the transition probabilities for each regime state
    """

    df = port_returns.to_frame("return").join(labels_df["label"], how="inner") # merge them on an inner join to align indices

    today = df['label']
    tomorrow = df['label'].shift(-1)
    transition_matrix = pd.crosstab(today, tomorrow)
    transition_probs = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

    return transition_probs


def compute_regime_feature_profile(X: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean value of each classifying FEATURE (not return) per regime -- what
    the regime looks like in terms of the raw inputs used to fit it, e.g.
    "regime 2 is the only one with vol_short_long_ratio > 1."

    Complements compute_regime_stats, which summarizes RETURNS observed
    while in each regime -- this instead summarizes the features that
    defined the regime in the first place.
    """
    merged = X.join(labels_df["label"], how="inner")
    return merged.groupby("label").mean()


def plot_regime_timeline(port_r: pd.Series, labels_df: pd.DataFrame, regime_stats: pd.DataFrame) -> None:
    """
    Plot daily portfolio returns over time, with the background shaded by
    which regime was active on each date -- the standard way regime-
    detection results get visually communicated. Contiguous runs of the
    same regime are shaded as single blocks, not per-day slivers.

    Regime->color/name mapping is derived from regime_stats (lowest std =
    calm, highest std = crisis, remaining = transitional), NOT hardcoded
    indices -- GMM's regime numbering is arbitrary per fit.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    returns_aligned = port_r.loc[labels_df.index]
    labels = labels_df["label"]

    calm_regime = int(regime_stats["std"].idxmin())
    crisis_regime = int(regime_stats["std"].idxmax())
    other_regimes = [r for r in regime_stats.index if r not in (calm_regime, crisis_regime)]

    color_map = {calm_regime: "#7e9ad7", crisis_regime: "#e57373"}
    name_map = {calm_regime: "Calm", crisis_regime: "Crisis"}
    for r in other_regimes:
        color_map[r] = "#ffcc80"
        name_map[r] = "Transitional"

    fig, ax = plt.subplots(figsize=(14, 6))

    # Shade contiguous regime runs as single blocks
    run_id = labels.ne(labels.shift()).cumsum()
    for _, run in labels.groupby(run_id):
        ax.axvspan(run.index[0], run.index[-1], color=color_map[int(run.iloc[0])], alpha=0.35, linewidth=0)

    ax.plot(returns_aligned.index, returns_aligned.values, color="black", linewidth=0.6)
    ax.axhline(0, color="gray", linewidth=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily portfolio return")
    ax.set_title(f"Daily portfolio returns ({fetch_start}--{fetch_end}) with GMM regime states")

    handles = [
        Patch(facecolor=color_map[r], alpha=0.35, label=name_map[r])
        for r in [calm_regime, *other_regimes, crisis_regime]
    ]
    ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    plt.show()


def main():
    price_hist = fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    prices = price_hist.prices
    asset_returns = prices_to_returns(prices)
    port_r = portfolio_returns(asset_returns=asset_returns, weights=weights)

    # convert these to DataFrames (those signatures should change)
    port_p_df = portfolio_value_series(prices, weights).to_frame(name="portfolio")
    port_r_df = port_r.to_frame(name="portfolio")

    feature_frames = build_feature_frames(port_p_df, port_r_df)
    X = build_feature_matrix(feature_frames).dropna()               # drop the NaN's or the GMM will complain
    model, scaler, X_scaled = fit_regime_model(X, 3, "full", 42)

    df = label_regimes(model, X_scaled, X.index)

    stats = compute_regime_stats(port_returns=port_r, labels_df=df)
    transition_probs = compute_transition_probs(port_returns=port_r, labels_df=df)

    print(transition_probs)

    feature_profile = compute_regime_feature_profile(X, df)
    print(feature_profile)

    #plot_regime_timeline(port_r, df, stats)

    #print(df.groupby(['label'])['label'].value_counts())

    #print(df.loc['2020-02-20':'2020-04-07'])
    #print(df.shape)

if __name__ == "__main__":
    main()
