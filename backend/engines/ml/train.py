from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns

from engines.features.volatility_features import rolling_volatility, ewm_volatility
from engines.features.drawdown_features import drawdown
from engines.features.absolute_returns_features import ewma_abs_returns, rolling_abs_returns
from engines.features.moving_average_features import moving_average_ratio
from engines.features.momentum_features import momentum_swings
from engines.targets.future_realized_volatility import future_realized_volatility
from engines.ml.dataset_builder import assemble_dataset
from engines.ml.cross_validation import run_rolling_cv
import matplotlib.pyplot as plt
#Training data
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
fetch_start = "2017-03-01"
fetch_end = "2025-03-01"

def main():
    """
    
    """
    #Data Ingestion
    price_history = fetch_price_history(tickers=tickers,start=fetch_start,end=fetch_end)
    prices = price_history.prices
    #Fetch price data checkpoint
    # print(prices.head())
    # print(prices.shape)
    # print(prices.index.min(), prices.index.max())
    # print(prices.columns.tolist())

    #Convert prices to returns
    returns = prices_to_returns(prices)
    print(returns.head())
    print(returns.shape)

    #Build features dict
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
        # "abs_interaction":ewma_abs_df*rolling_abs_20_df,
        # "vol_interaction":ewm_vol_df*ewma_abs_df,
        # "regime_interaction":ma_ratio_df*drawdown_df,
        # "stress_vol_interaction":ewma_abs_df*drawdown_df,
        # "trend_vol_interaction":ewma_abs_df*ma_ratio_df,
        # "vol_term_structure_interaction":rolling_vol_20_df*rolling_abs_20_df,
    }
    
    #Build target dataframe
    target_df = future_realized_volatility(returns, horizon=10, annualize=True)
    
    #Now, build dataset for split
    X,y = assemble_dataset(features, target_df)
    # print("X type:", type(X))
    # print("y type:", type(y))
    # print("X shape:", X.shape)
    # print("y shape:", y.shape)

    # print("\nX preview:")
    # print(X.tail())

    # print("\ny preview:")
    # print(y.tail())

    #Train model
    results, preds = run_rolling_cv(X,y,lambda: Ridge(alpha=1.0),start_date=fetch_start,end_date=fetch_end,train_years=4,validation_years=1,baseline_column="ewm_volatility",return_predictions=True)
    print(results[[
        "fold",
        "val_rmse",
        "baseline_val_rmse",
        "rmse_improvement_vs_baseline",
        "val_mae",
        "baseline_val_mae",
        "mae_improvement_vs_baseline",
    ]])
    print(results[["train_rmse", "val_rmse", "train_mae", "val_mae"]].mean())

    ticker_df = preds[preds["ticker"] == "AAPL"].sort_values("date")

    plt.figure(figsize=(12, 6))
    plt.plot(ticker_df["date"], ticker_df["y_true"], label="Actual")
    plt.plot(ticker_df["date"], ticker_df["y_pred"], label="Predicted", linestyle="--")
    plt.plot(ticker_df["date"], ticker_df["baseline_pred"], label="EWMA Baseline", linestyle=":")

    plt.title("NVDA: Actual vs Predicted Future Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()