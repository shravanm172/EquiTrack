"""
Rolling walk-forward backtest for a GARCH(1,1) + ML residual-correction
hybrid volatility model.

Ridge (same feature set as asset_vol_predictor.py) predicts the leftover
error GARCH's own forecast doesn't explain:

    log_ratio_target = log(actual_future_vol / garch_forecast_vol)

Recombined at inference:

    hybrid_pred = garch_forecast_vol * exp(predicted_log_ratio)

If Ridge learns nothing useful, predicted_log_ratio -> 0 and
hybrid_pred -> garch_forecast_vol, so the hybrid can never structurally do
worse than plain GARCH -- it can only add value on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns
from engines.features.volatility_features import rolling_volatility
from engines.targets.future_realized_volatility import future_realized_volatility
from engines.ml.models.asset_vol_predictor import build_feature_frames
from engines.garch_engine import calibrate_garch_mle, GarchCalibratedParams

# Config
h = 30
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
fetch_start = "2017-03-01"
fetch_end = "2026-03-23"
MIN_HISTORY = 252 + 5 + 1 + h


def _compute_filtered_variances(
    returns: np.ndarray,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    h0: float,
) -> np.ndarray:
    """In-sample filtered GARCH(1,1) variance path -- one value per observed day."""
    n = len(returns)
    h_path = np.empty(n, dtype=float)
    h_path[0] = max(h0, 1e-12)
    for t in range(1, n):
        eps_prev = returns[t - 1] - mu
        h_path[t] = omega + alpha * (eps_prev ** 2) + beta * h_path[t - 1]
        h_path[t] = max(h_path[t], 1e-12)
    return h_path


def garch_forecast_vol_path(
    omega: float,
    alpha: float,
    beta: float,
    h_path: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Closed-form horizon-day-ahead GARCH(1,1) vol forecast, originating from
    every date in h_path at once (vectorized -- see chat for the derivation).
    """
    phi = alpha + beta
    uncond_var = omega / (1 - phi)

    k = np.arange(1, horizon + 1)
    weight = np.mean(phi ** k)

    avg_h = uncond_var + (h_path - uncond_var) * weight
    return np.sqrt(avg_h) * np.sqrt(252)


def build_garch_forecast_series(returns_history: pd.Series, horizon: int) -> pd.Series:
    """
    Fit GARCH once on this ticker's history-to-date, forecast horizon-ahead
    vol originating from every date in that history.
    """
    r = returns_history.to_numpy(dtype=float)
    params: GarchCalibratedParams = calibrate_garch_mle(returns_history, estimate_mu=False)
    h_path = _compute_filtered_variances(
        r, params.mu, params.omega, params.alpha, params.beta, params.h0
    )
    forecast_vol = garch_forecast_vol_path(params.omega, params.alpha, params.beta, h_path, horizon)
    return pd.Series(forecast_vol, index=returns_history.index)


def main():
    # Data ingestion
    price_hist = fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    prices = price_hist.prices
    returns = prices_to_returns(prices)

    # Future realized vol (what we ultimately want to predict)
    target_df = future_realized_volatility(returns, horizon=h, annualize=True)

    # Trailing 30d vol, kept only as a third comparison baseline
    baseline_vol_df = rolling_volatility(returns, window=30, annualize=True)

    results = []

    for i in range(MIN_HISTORY, len(returns) - h, 20):
        forecast_date = returns.index[i]

        # Training context available up to forecast_date (no leakage)
        context_prices = prices.loc[:forecast_date]
        context_returns = returns.iloc[:i + 1]

        # Same engineered features as before
        feature_frames = build_feature_frames(context_prices, context_returns)

        # One GARCH fit per ticker per fold, forecast-vol at every historical
        # date in this fold's context.
        garch_forecast_by_ticker: dict[str, pd.Series] = {}
        for ticker in tickers:
            garch_forecast_by_ticker[ticker] = build_garch_forecast_series(
                context_returns[ticker], horizon=h
            )

        rows = []
        for date in context_returns.index:
            for ticker in tickers:
                row = {"date": date, "ticker": ticker}
                for feat_name, feat_df in feature_frames.items():
                    if date in feat_df.index and ticker in feat_df.columns:
                        row[feat_name] = feat_df.loc[date, ticker]
                    else:
                        row[feat_name] = np.nan

                # Residual/log-ratio training target:
                #     log(actual_future_vol / garch_forecast_vol)
                actual_vol = (
                    target_df.loc[date, ticker]
                    if date in target_df.index and ticker in target_df.columns
                    else np.nan
                )
                garch_vol = garch_forecast_by_ticker[ticker].get(date, np.nan)
                if pd.notna(actual_vol) and pd.notna(garch_vol) and garch_vol > 0:
                    row["log_ratio_target"] = np.log(actual_vol / garch_vol)
                else:
                    row["log_ratio_target"] = np.nan
                rows.append(row)

        train_df = pd.DataFrame(rows).dropna()
        feature_cols = [
            c for c in train_df.columns if c not in ("date", "ticker", "log_ratio_target")
        ]

        X_train = train_df[feature_cols]
        y_train = train_df["log_ratio_target"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Predict for each ticker at forecast_date
        for ticker in tickers:
            row = {}
            for feat_name, feat_df in feature_frames.items():
                row[feat_name] = feat_df.loc[forecast_date, ticker]

            X_latest = pd.DataFrame([row])[feature_cols]
            X_latest_scaled = scaler.transform(X_latest)

            predicted_log_ratio = float(model.predict(X_latest_scaled)[0])
            garch_pred = float(garch_forecast_by_ticker[ticker].loc[forecast_date])

            # Recombine GARCH's forecast with the ML residual correction.
            hybrid_pred = garch_pred * np.exp(predicted_log_ratio)

            baseline_pred = float(baseline_vol_df.loc[forecast_date, ticker])
            actual = float(target_df.loc[forecast_date, ticker])

            results.append({
                "date": forecast_date,
                "ticker": ticker,
                "actual": actual,
                "hybrid_pred": hybrid_pred,
                "garch_pred": garch_pred,
                "baseline_pred": baseline_pred,
                "predicted_log_ratio": predicted_log_ratio,
            })

    # Evaluation
    df = pd.DataFrame(results)

    print("\n=== GARCH + ML Residual Hybrid Backtest Results ===")
    print("Rows:", len(df))
    print("Start:", df["date"].min())
    print("End:  ", df["date"].max())

    hybrid_mae = mean_absolute_error(df["actual"], df["hybrid_pred"])
    garch_mae = mean_absolute_error(df["actual"], df["garch_pred"])
    baseline_mae = mean_absolute_error(df["actual"], df["baseline_pred"])
    hybrid_rmse = np.sqrt(mean_squared_error(df["actual"], df["hybrid_pred"]))
    garch_rmse = np.sqrt(mean_squared_error(df["actual"], df["garch_pred"]))
    baseline_rmse = np.sqrt(mean_squared_error(df["actual"], df["baseline_pred"]))

    print("\nMAE")
    print("Hybrid:  ", hybrid_mae)
    print("GARCH:   ", garch_mae)
    print("Baseline:", baseline_mae)

    print("\nRMSE")
    print("Hybrid:  ", hybrid_rmse)
    print("GARCH:   ", garch_rmse)
    print("Baseline:", baseline_rmse)

    mae_improvement_vs_garch = (garch_mae - hybrid_mae) / garch_mae
    rmse_improvement_vs_garch = (garch_rmse - hybrid_rmse) / garch_rmse

    print("\nImprovement vs plain GARCH  <-- this is the number that matters")
    print(f"MAE improvement:  {mae_improvement_vs_garch:.2%}")
    print(f"RMSE improvement: {rmse_improvement_vs_garch:.2%}")

    df["vol_regime"] = pd.qcut(df["actual"], q=3, labels=["calm", "medium", "stress"])

    print("\nRegime MAE")
    for regime, subdf in df.groupby("vol_regime"):
        hybrid_mae_reg = mean_absolute_error(subdf["actual"], subdf["hybrid_pred"])
        garch_mae_reg = mean_absolute_error(subdf["actual"], subdf["garch_pred"])
        imp = (garch_mae_reg - hybrid_mae_reg) / garch_mae_reg
        print(f"{regime}: Hybrid={hybrid_mae_reg:.6f}, GARCH={garch_mae_reg:.6f}, Hybrid vs GARCH={imp:.2%}")


if __name__ == "__main__":
    main()
