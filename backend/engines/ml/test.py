from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns
from backend.engines.ml.models.asset_vol_predictor import build_feature_frames
from engines.targets.future_realized_volatility import future_realized_volatility
from engines.ml.dataset_builder import assemble_dataset
import matplotlib.pyplot as plt



tickers_train = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
tickers_test = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]

train_start = "2017-03-01"
train_end = "2025-03-01"


test_start = "2025-03-01"
test_eval_start = "2024-12-01" #provides buffered history to compute test set features
test_end = "2026-03-01"

def main():
    """
    Retrain the model that we chose from the rolling cv phase on the full training+validation dataset 
    Test this new fitted model on the test set and evaluate 
    """
    #Data Ingestion for training set
    train_price_history = fetch_price_history(tickers=tickers_train,start=train_start,end=train_end)
    train_prices = train_price_history.prices
    #Convert prices to returns
    train_returns = prices_to_returns(train_prices)
    print(train_returns.head())
    print(train_returns.shape)
    
    #Build features dict for training set
    train_features = build_feature_frames(prices=train_prices, returns=train_returns)
    
    #Build target dataframe
    train_target_df = future_realized_volatility(train_returns, horizon=30, annualize=True)

    X_train, y_train = assemble_dataset(train_features, train_target_df)
    #Scale training data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    #Data ingestion for test set
    test_price_history = fetch_price_history(tickers=tickers_test, start=test_eval_start,end=test_end)
    test_prices = test_price_history.prices
    test_returns = prices_to_returns(test_prices)
    print(test_returns.head())
    print(test_returns.shape)

    #Build features dict for test set
    test_features = build_feature_frames(prices=test_prices, returns=test_returns)
    #Build target dataframe
    test_target_df = future_realized_volatility(test_returns, horizon=10, annualize=True)
    
    #Filter out the dates outside the test set window
    X_test_full, y_test_full = assemble_dataset(test_features, test_target_df)
    mask = X_test_full.index.get_level_values(0) >= test_start
    X_test = X_test_full.loc[mask]
    y_test = y_test_full.loc[mask]

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )

    y_test_pred = model.predict(X_test_scaled)
    baseline_pred = X_test["ewm_volatility"].to_numpy()

    #Evaluation
    ridge_mse = mean_squared_error(y_test, y_test_pred)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_mae = mean_absolute_error(y_test, y_test_pred)
    ridge_r2 = r2_score(y_test, y_test_pred)

    baseline_mse = mean_squared_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)

    print("\n=== Test Set Performance ===")
    print(f"Test rows: {len(X_test)}")
    print(f"Test start: {X_test.index.get_level_values(0).min()}")
    print(f"Test end:   {X_test.index.get_level_values(0).max()}")

    print("\nRidge Model:")
    print(f"  MSE:  {ridge_mse:.6f}")
    print(f"  RMSE: {ridge_rmse:.6f}")
    print(f"  MAE:  {ridge_mae:.6f}")
    print(f"  R^2:  {ridge_r2:.6f}")

    print("\nBaseline (ewm_volatility):")
    print(f"  MSE:  {baseline_mse:.6f}")
    print(f"  RMSE: {baseline_rmse:.6f}")
    print(f"  MAE:  {baseline_mae:.6f}")
    print(f"  R^2:  {baseline_r2:.6f}")

    results_df = pd.DataFrame({
        "actual": y_test,
        "ridge_pred": y_test_pred,
        "baseline_pred": baseline_pred,
    }, index=y_test.index)

    print("\nRandom sample predictions:")
    print(results_df.sample(10, random_state=42))

    improvement = (baseline_mse - ridge_mse) / baseline_mse
    print(f"Improvement in mse: {improvement:.2%}")

    # coef_df = pd.Series(model.coef_, index=X_train.columns)
    # print("\nMost important coefficients:")
    # print(coef_df.sort_values(key=abs, ascending=False))

    #Plot baseline vs predicted future realized volatility
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index.get_level_values(0), results_df["actual"], label="Actual")
    plt.plot(results_df.index.get_level_values(0), results_df["ridge_pred"], label="Ridge Prediction")
    plt.plot(results_df.index.get_level_values(0), results_df["baseline_pred"], label="Baseline (EWM Vol)")
    plt.legend()
    plt.title("Test Set: Actual vs Predicted Future Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()