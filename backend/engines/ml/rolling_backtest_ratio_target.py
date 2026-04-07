"""
Rolling backtest for ML log-ratio-target vol predictor.
Instead of predicting future realized vol directly, the model predicts:
    predicted_ratio = log(future_realized_vol / current_ewma_vol)

At inference the final prediction is:
    vol_pred = predicted_ratio * ewma_vol
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns
from engines.features.volatility_features import ewm_volatility, rolling_volatility
from engines.targets.future_realized_volatility import future_realized_volatility
from engines.ml.models.asset_vol_predictor import build_feature_frames

# Config
h = 30
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
fetch_start = "2017-03-01"
fetch_end = "2026-03-23"


def main():
    # Data ingestion
    price_hist = fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    prices = price_hist.prices
    returns = prices_to_returns(prices)

    # Future realized vol (what we ultimately want to predict)
    target_df = future_realized_volatility(returns, horizon=h, annualize=True)

    # 30-day trailing realized vol — baseline and ratio denominator
    baseline_vol_df = rolling_volatility(returns, window=30, annualize=True)

    # HAR components: daily (2d), weekly (5d), monthly (22d) realized vol
    rv_daily_df = rolling_volatility(returns, window=2, annualize=True)
    rv_weekly_df = rolling_volatility(returns, window=5, annualize=True)
    rv_monthly_df = rolling_volatility(returns, window=22, annualize=True)

    # Build log-ratio target
    # log_ratio = log(future_realized_vol / trailing_30d_vol)
    # Log compresses right-tail outliers so stress periods don't dominate the squared-error loss and bias predictions upward.
    ratio_target_df = np.log(target_df / baseline_vol_df)

    results = []
    MIN_HISTORY = 252 + 5 + 1 + h

    # Rolling walk-forward loop, step forward by 20 
    for i in range(MIN_HISTORY, len(returns) - h, 20):
        forecast_date = returns.index[i]

        # Training context available up to forecast_date (no leakage)
        context_prices = prices.loc[:forecast_date]
        context_returns = returns.iloc[:i + 1]

        # Compute features 
        feature_frames = build_feature_frames(context_prices, context_returns)

        # Build stacked training set 
        rows = []
        for date in context_returns.index:
            for ticker in tickers:
                row = {"date": date, "ticker": ticker}
                for feat_name, feat_df in feature_frames.items():
                    if date in feat_df.index and ticker in feat_df.columns:
                        row[feat_name] = feat_df.loc[date, ticker]
                    else:
                        row[feat_name] = np.nan
                # Ratio target for training
                if date in ratio_target_df.index and ticker in ratio_target_df.columns:
                    row["ratio_target"] = ratio_target_df.loc[date, ticker]
                else:
                    row["ratio_target"] = np.nan
                rows.append(row)

        train_df = pd.DataFrame(rows).dropna()
        feature_cols = [c for c in train_df.columns if c not in ("date", "ticker", "ratio_target")]

        X_train = train_df[feature_cols]
        y_train = train_df["ratio_target"]

        # Scale & fit
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Predict for each ticker at forecast_date

        # HAR: fit on training window
        har_rows = []
        for date in context_returns.index:
            for ticker in tickers:
                if (date in rv_daily_df.index and date in target_df.index
                        and ticker in rv_daily_df.columns):
                    rv_d = rv_daily_df.loc[date, ticker]
                    rv_w = rv_weekly_df.loc[date, ticker]
                    rv_m = rv_monthly_df.loc[date, ticker]
                    y_val = target_df.loc[date, ticker]
                    if pd.notna(rv_d) and pd.notna(rv_w) and pd.notna(rv_m) and pd.notna(y_val):
                        har_rows.append({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m, "target": y_val})

        har_train_df = pd.DataFrame(har_rows)
        har_model = LinearRegression()
        har_model.fit(har_train_df[["rv_d", "rv_w", "rv_m"]], har_train_df["target"])

        for ticker in tickers:
            row = {}
            for feat_name, feat_df in feature_frames.items():
                row[feat_name] = feat_df.loc[forecast_date, ticker]

            X_latest = pd.DataFrame([row])[feature_cols]
            X_latest_scaled = scaler.transform(X_latest)

            predicted_log_ratio = float(model.predict(X_latest_scaled)[0])

            # Convert log-ratio prediction back to vol: exp(log_ratio) * trailing_30d_vol
            baseline_now = float(baseline_vol_df.loc[forecast_date, ticker])
            predicted_ratio = np.exp(predicted_log_ratio)
            ml_pred = predicted_ratio * baseline_now

            baseline_pred = baseline_now
            actual = float(target_df.loc[forecast_date, ticker])

            # HAR's prediction (baseline pred)
            har_input = pd.DataFrame([{
                "rv_d": rv_daily_df.loc[forecast_date, ticker],
                "rv_w": rv_weekly_df.loc[forecast_date, ticker],
                "rv_m": rv_monthly_df.loc[forecast_date, ticker],
            }])
            har_pred = float(har_model.predict(har_input)[0])

            results.append({
                "date": forecast_date,
                "ticker": ticker,
                "actual": actual,
                "ml_pred": ml_pred,
                "baseline_pred": baseline_pred,
                "har_pred": har_pred,
                "predicted_ratio": predicted_ratio,
            })

    # Evaluation
    df = pd.DataFrame(results)

    print("\n=== Rolling Ratio-Target Backtest Results ===")
    print("Rows:", len(df))
    print("Start:", df["date"].min())
    print("End:  ", df["date"].max())

    ml_mae = mean_absolute_error(df["actual"], df["ml_pred"])
    baseline_mae = mean_absolute_error(df["actual"], df["baseline_pred"])
    har_mae = mean_absolute_error(df["actual"], df["har_pred"])
    ml_rmse = np.sqrt(mean_squared_error(df["actual"], df["ml_pred"]))
    baseline_rmse = np.sqrt(mean_squared_error(df["actual"], df["baseline_pred"]))
    har_rmse = np.sqrt(mean_squared_error(df["actual"], df["har_pred"]))

    print("\nMAE")
    print("ML:      ", ml_mae)
    print("HAR:     ", har_mae)
    print("Baseline:", baseline_mae)

    print("\nRMSE")
    print("ML:      ", ml_rmse)
    print("HAR:     ", har_rmse)
    print("Baseline:", baseline_rmse)

    df["err_ml"] = df["ml_pred"] - df["actual"]
    df["err_baseline"] = df["baseline_pred"] - df["actual"]
    df["err_har"] = df["har_pred"] - df["actual"]

    print("\nBias")
    print("ML mean error:      ", df["err_ml"].mean())
    print("HAR mean error:     ", df["err_har"].mean())
    print("Baseline mean error:", df["err_baseline"].mean())

    mae_improvement_vs_har = (har_mae - ml_mae) / har_mae
    rmse_improvement_vs_har = (har_rmse - ml_rmse) / har_rmse
    mae_improvement = (baseline_mae - ml_mae) / baseline_mae
    rmse_improvement = (baseline_rmse - ml_rmse) / baseline_rmse

    print("\nImprovement vs Baseline (30d trailing vol)")
    print(f"MAE improvement:  {mae_improvement:.2%}")
    print(f"RMSE improvement: {rmse_improvement:.2%}")

    print("\nImprovement vs HAR")
    print(f"MAE improvement:  {mae_improvement_vs_har:.2%}")
    print(f"RMSE improvement: {rmse_improvement_vs_har:.2%}")

    print("\nRatio stats")
    print(f"Mean predicted ratio:   {df['predicted_ratio'].mean():.4f}")
    print(f"Median predicted ratio: {df['predicted_ratio'].median():.4f}")

    df["abs_err_ml"] = (df["ml_pred"] - df["actual"]).abs()
    df["abs_err_baseline"] = (df["baseline_pred"] - df["actual"]).abs()

    df["vol_regime"] = pd.qcut(
        df["actual"],
        q=3,
        labels=["calm", "medium", "stress"]
    )

    print("\nRegime MAE")
    for regime, subdf in df.groupby("vol_regime"):
        ml_mae_reg = mean_absolute_error(subdf["actual"], subdf["ml_pred"])
        har_mae_reg = mean_absolute_error(subdf["actual"], subdf["har_pred"])
        base_mae_reg = mean_absolute_error(subdf["actual"], subdf["baseline_pred"])
        imp_vs_har = (har_mae_reg - ml_mae_reg) / har_mae_reg
        imp_vs_base = (base_mae_reg - ml_mae_reg) / base_mae_reg
        print(
            f"{regime}: "
            f"ML={ml_mae_reg:.6f}, "
            f"HAR={har_mae_reg:.6f}, "
            f"Baseline={base_mae_reg:.6f}, "
            f"ML vs HAR={imp_vs_har:.2%}, "
            f"ML vs Base={imp_vs_base:.2%}"
        )


if __name__ == "__main__":
    main()
