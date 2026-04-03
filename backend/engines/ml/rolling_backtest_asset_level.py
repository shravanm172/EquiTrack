from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns
from engines.ml.feature_scaling import scale_fold_features
from engines.targets.future_realized_volatility import future_realized_volatility
from backend.engines.ml.models.asset_vol_predictor import build_models, build_feature_frames


h= 30
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
fetch_start = "2017-03-01"
fetch_end = "2026-03-23"

def main():
    price_hist = fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    prices = price_hist.prices
    returns = prices_to_returns(prices)
    target_df = future_realized_volatility(returns, horizon=h, annualize=True)
    results = []

    #ensure enough history in range to support features computations 
    MIN_HISTORY = 252 + 5 + 1 + h  # rolling + lag + diff + horizon buffer
    for i in range(MIN_HISTORY, len(returns) - h, 20):
        #Split the training context from the forecast horizon
        forecast_date = returns.index[i]
        train_end = (forecast_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        #Train model
        models = build_models(horizons=[h], tickers=tickers, fetch_start=fetch_start, fetch_end=train_end)
        bundle = models[h]
        #Make prediction
        #Compute input features for prediction
        #Need context prices and returns
        context_prices = prices.loc[:forecast_date].copy()
        context_returns = returns.iloc[:i+1].copy()
        feature_frames = build_feature_frames(context_prices, context_returns)
        
        #scale using scaler from bundle
        for ticker in tickers:
            row = {}

            for feature_name, df in feature_frames.items():
                if ticker not in df.columns:
                    raise ValueError(f"{ticker} missing from feature frame {feature_name}")
                row[feature_name] = df.loc[forecast_date, ticker]

            X_latest = pd.DataFrame([row])
            X_latest = X_latest[bundle.feature_columns]

            X_latest_scaled = pd.DataFrame(
                bundle.scaler.transform(X_latest),
                columns=X_latest.columns,
            )

            ml_pred = float(bundle.model.predict(X_latest_scaled)[0])
            baseline_pred = float(feature_frames["ewm_volatility"].loc[forecast_date, ticker])
            actual = float(target_df.loc[forecast_date, ticker])
        
        
            results.append({
                "date": forecast_date,
                "ticker": ticker,
                "actual": actual,
                "ml_pred": ml_pred,
                "baseline_pred": baseline_pred,
            })
        
    df = pd.DataFrame(results)

    print("\n=== Rolling Asset-Level Backtest Results ===")
    print("Rows:", len(df))
    print("Start:", df["date"].min())
    print("End:  ", df["date"].max())

    ml_mae = mean_absolute_error(df["actual"], df["ml_pred"])
    baseline_mae = mean_absolute_error(df["actual"], df["baseline_pred"])

    ml_rmse = np.sqrt(mean_squared_error(df["actual"], df["ml_pred"]))
    baseline_rmse = np.sqrt(mean_squared_error(df["actual"], df["baseline_pred"]))

    print("\nMAE")
    print("ML:      ", ml_mae)
    print("Baseline:", baseline_mae)

    print("\nRMSE")
    print("ML:      ", ml_rmse)
    print("Baseline:", baseline_rmse)

    df["err_ml"] = df["ml_pred"] - df["actual"]
    df["err_baseline"] = df["baseline_pred"] - df["actual"]

    print("\nBias")
    print("ML mean error:      ", df["err_ml"].mean())
    print("Baseline mean error:", df["err_baseline"].mean())

    mae_improvement = (baseline_mae - ml_mae) / baseline_mae
    rmse_improvement = (baseline_rmse - ml_rmse) / baseline_rmse

    print("\nImprovement")
    print(f"MAE improvement:  {mae_improvement:.2%}")
    print(f"RMSE improvement: {rmse_improvement:.2%}")

    # print("\nPer-ticker MAE")
    # for ticker, subdf in df.groupby("ticker"):
    #     t_ml_mae = mean_absolute_error(subdf["actual"], subdf["ml_pred"])
    #     t_baseline_mae = mean_absolute_error(subdf["actual"], subdf["baseline_pred"])
    #     t_improvement = (t_baseline_mae - t_ml_mae) / t_baseline_mae
    #     print(
    #         f"{ticker}: "
    #         f"ML={t_ml_mae:.6f}, "
    #         f"Baseline={t_baseline_mae:.6f}, "
    #         f"Improvement={t_improvement:.2%}"
    #     )

    # print("\nPer-ticker RMSE")
    # for ticker, subdf in df.groupby("ticker"):
    #     t_ml_rmse = np.sqrt(mean_squared_error(subdf["actual"], subdf["ml_pred"]))
    #     t_baseline_rmse = np.sqrt(mean_squared_error(subdf["actual"], subdf["baseline_pred"]))
    #     t_improvement = (t_baseline_rmse - t_ml_rmse) / t_baseline_rmse
    #     print(
    #         f"{ticker}: "
    #         f"ML={t_ml_rmse:.6f}, "
    #         f"Baseline={t_baseline_rmse:.6f}, "
    #         f"Improvement={t_improvement:.2%}"
    #     )

    # print("\nPer-ticker Bias")
    # for ticker, subdf in df.groupby("ticker"):
    #     print(
    #         f"{ticker}: "
    #         f"ML bias={subdf['err_ml'].mean():.6f}, "
    #         f"Baseline bias={subdf['err_baseline'].mean():.6f}"
    #     )

    # df["abs_err_ml"] = (df["ml_pred"] - df["actual"]).abs()
    # df["abs_err_baseline"] = (df["baseline_pred"] - df["actual"]).abs()

    # df["vol_regime"] = pd.qcut(
    #     df["actual"],
    #     q=3,
    #     labels=["calm", "medium", "stress"]
    # )

    print("\nRegime MAE")
    for regime, subdf in df.groupby("vol_regime"):
        ml_mae_reg = mean_absolute_error(subdf["actual"], subdf["ml_pred"])
        base_mae_reg = mean_absolute_error(subdf["actual"], subdf["baseline_pred"])
        improvement = (base_mae_reg - ml_mae_reg) / base_mae_reg
        print(
            f"{regime}: "
            f"ML={ml_mae_reg:.6f}, "
            f"Baseline={base_mae_reg:.6f}, "
            f"Improvement={improvement:.2%}"
        )

if __name__ == "__main__":
    main()