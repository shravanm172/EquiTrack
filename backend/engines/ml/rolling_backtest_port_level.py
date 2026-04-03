import pandas as pd
import numpy as np
from engines.portfolio_engine import (
    portfolio_returns,
    prices_to_returns,
    portfolio_volatility,
    covariance_matrix,
)
from providers.market_data import fetch_price_history
from engines.forecast_estimators import estimate_volatility
from backend.engines.ml.models.asset_vol_predictor import build_models, build_feature_frames

h = 30
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
weights = {
    "AAPL": 0.2,
    "MSFT": 0.2,
    "AMZN": 0.2,
    "GOOGL": 0.2,
    "NVDA": 0.2,
}
fetch_start = "2017-03-01"
fetch_end = "2026-03-23"


def main():
    price_history = fetch_price_history(tickers, start=fetch_start, end=fetch_end)
    prices = price_history.prices
    returns = prices_to_returns(prices)

    results = []

    # start at 100 just to ensure enough history for features / rolling windows
    for i in range(252, len(returns) - h, 20):
        returns_context = returns.iloc[:i].copy()
        returns_holdout = returns.iloc[i:i+h].copy()

        prices_context = prices.loc[:returns_context.index[-1]].copy()

        forecast_origin = returns_context.index[-1]

        # ===== Ground truth =====
        port_r_holdout = portfolio_returns(returns_holdout, weights)
        realized_vol = float(port_r_holdout.std(ddof=1))

        # ===== Baselines =====
        port_r_context = portfolio_returns(returns_context, weights)

        rolling_vol, _ = estimate_volatility(
            port_r_context,
            mode="rolling",
            window=60,
        )
        ewma_vol, _ = estimate_volatility(
            port_r_context,
            mode="ewma",
        )

        # ===== ML model trained only up to this forecast origin =====
        train_end_for_model = (forecast_origin + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        models = build_models(
            horizons=[h],
            tickers=tickers,
            fetch_start=fetch_start,
            fetch_end=train_end_for_model,
        )
        bundle = models[h]

        # ===== ML prediction at forecast origin =====
        feature_frames = build_feature_frames(prices_context, returns_context)

        predicted_vols_daily = {}

        for ticker in tickers:
            row = {}

            for feature_name, df in feature_frames.items():
                if ticker not in df.columns:
                    raise ValueError(f"{ticker} missing from feature frame {feature_name}")
                row[feature_name] = df.loc[forecast_origin, ticker]

            X_latest = pd.DataFrame([row])
            X_latest = X_latest[bundle.feature_columns]

            X_latest_scaled = pd.DataFrame(
                bundle.scaler.transform(X_latest),
                columns=X_latest.columns,
            )

            pred_annual = float(bundle.model.predict(X_latest_scaled)[0])
            predicted_vols_daily[ticker] = pred_annual / np.sqrt(252)

        cov_df = covariance_matrix(returns_context, predicted_vols_daily, weights, corr_method="ewma")
        ml_vol = float(portfolio_volatility(cov_df, weights))

        results.append({
            "date": forecast_origin,
            "realized": realized_vol,
            "ml": ml_vol,
            "rolling": float(rolling_vol),
            "ewma": float(ewma_vol),
        })

    df = pd.DataFrame(results)

    print("\n=== Rolling Backtest Results ===")
    print("Rows:", len(df))
    print("Start:", df["date"].iloc[0])
    print("End:  ", df["date"].iloc[-1])

    print("\nMAE")
    print("ML:     ", (df["ml"] - df["realized"]).abs().mean())
    print("Rolling:", (df["rolling"] - df["realized"]).abs().mean())
    print("EWMA:   ", (df["ewma"] - df["realized"]).abs().mean())

    print("\nRMSE")
    print("ML:     ", np.sqrt(((df["ml"] - df["realized"]) ** 2).mean()))
    print("Rolling:", np.sqrt(((df["rolling"] - df["realized"]) ** 2).mean()))
    print("EWMA:   ", np.sqrt(((df["ewma"] - df["realized"]) ** 2).mean()))

    df.plot(x="date", y=["realized", "ml", "ewma", "rolling"])


if __name__ == "__main__":
    main()