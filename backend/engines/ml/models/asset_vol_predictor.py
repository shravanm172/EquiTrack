from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns
from engines.features.volatility_features import rolling_volatility, ewm_volatility
from engines.features.drawdown_features import drawdown
from engines.features.absolute_returns_features import ewma_abs_returns, rolling_abs_returns
from engines.features.moving_average_features import moving_average_ratio
from engines.features.momentum_features import momentum_swings
from engines.targets.future_realized_volatility import future_realized_volatility
from engines.ml.dataset_builder import assemble_dataset

default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"] #Default set of tickers used to train model
default_fetch_start = "2017-03-01" #This is the earliest training set start date that was used when choosing the model in the rolling cv stage
default_fetch_end = "2026-03-20" #Most recent available trading day

@dataclass 
class VolatilityModelBundle:
    horizon:int
    model:Ridge
    scaler:StandardScaler
    feature_columns:list[str]
    train_tickers:list[str]
    train_start:str
    train_end:str
    annualize_target:bool


def build_feature_frames(prices:pd.DataFrame, returns:pd.DataFrame)->dict[str, pd.DataFrame]:
    """
    Build the features dicts used by each model
    """
    rolling_vol_20_df = rolling_volatility(returns)
    ewm_vol_df = ewm_volatility(returns)
    drawdown_df = drawdown(prices)
    ewma_abs_df = ewma_abs_returns(returns)
    rolling_abs_20_df = rolling_abs_returns(returns)
    ma_ratio_df = moving_average_ratio(prices)
    momentum_df = momentum_swings(returns)

    features = {
        "rolling_vol_20":rolling_vol_20_df,
        "ewm_volatility":ewm_vol_df,
        "drawdown":drawdown_df,
        "ewma_abs_returns":ewma_abs_df,
        "rolling_abs_returns_20":rolling_abs_20_df,
        "moving_avg_ratio":ma_ratio_df,
        "momentum_swings":momentum_df,
    }
    return features


def build_models(
    horizons: Iterable[int],
    tickers: list[str] | None = None,
    fetch_start: str | None = None,
    fetch_end: str | None = None,
    
) -> dict[int, VolatilityModelBundle]:
    """
    Accepts list of horizons and trains models for each
    Returns list of trained models
    """
    if tickers is None:
        tickers = default_tickers 
    if fetch_end is None:
        fetch_end = default_fetch_end
    if fetch_start is None:
        fetch_start = default_fetch_start
    horizons = sorted({int(h) for h in horizons})
    if not horizons:
        raise ValueError("At least one horizon must be provided.")
    if any(h <= 0 for h in horizons):
        raise ValueError("All horizons must be positive integers.")
    
    #Data ingestion
    price_history = fetch_price_history(tickers, start=fetch_start, end=fetch_end)
    prices = price_history.prices
    returns = prices_to_returns(prices)

    
    #Shared features for all horizons
    features = build_feature_frames(prices=prices, returns=returns)

    # Stack multi-asset features into long format (date × ticker rows)
    # so that the portfolio-level dataset builder can handle them.
    stacked_features = {}
    for name, df in features.items():
        s = df.stack()
        s.name = name
        stacked_features[name] = s.to_frame()

    model_bundles: dict[int, VolatilityModelBundle] = {}
    for h in horizons:
        target_df = future_realized_volatility(returns, horizon=h, annualize=True)
        stacked_target = target_df.stack().to_frame(name="target")

        X_train, y_train = assemble_dataset(stacked_features, stacked_target)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns,
        )
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        bundle = VolatilityModelBundle(
            horizon=h,
            model=model,
            scaler=scaler,
            feature_columns=list(X_train.columns),
            train_tickers=list(tickers),
            train_start=fetch_start,
            train_end=fetch_end,
            annualize_target=True,
        )

        model_bundles[h] = bundle
    return model_bundles

