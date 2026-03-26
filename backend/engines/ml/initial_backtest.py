import pandas as pd
import numpy as np
from engines.portfolio_engine import portfolio_returns, prices_to_returns
from providers.market_data import fetch_price_history
from engines.forecast_estimators import estimate_volatility
from engines.ml.models import build_models, build_feature_frames
from engines.portfolio_engine import portfolio_volatility, covariance_matrix

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
    #generate portfolio returns dataframe
    price_history = fetch_price_history(tickers, start=fetch_start, end=fetch_end)
    prices = price_history.prices
    returns = prices_to_returns(prices)

    port_r = portfolio_returns(returns, weights)

    #Inspect portfolio returns
    print("\n=== Portfolio Returns (head) ===")
    print(port_r.head())

    print("\n=== Portfolio Returns (tail) ===")
    print(port_r.tail())

    print("\n=== Info ===")
    print("Length:", len(port_r))
    print("Start date:", port_r.index[0])
    print("End date:", port_r.index[-1])

    print("\n=== Basic Stats ===")
    print(port_r.describe())

    #Split into forecast set and training set
    returns_holdout = returns.iloc[-h:].copy()
    returns_context = returns.iloc[:-h].copy()
    prices_context = prices.iloc[:-h].copy()

    port_r_holdout = port_r.iloc[-h:].copy()
    port_r_context = port_r.iloc[:-h].copy()

    forecast_origin = port_r_context.index[-1]
    holdout_start = port_r_holdout.index[0]
    holdout_end = port_r_holdout.index[-1]

    print("\n=== Split Info ===")
    print("Forecast origin:", forecast_origin.date())
    print("Holdout start:  ", holdout_start.date())
    print("Holdout end:    ", holdout_end.date())
    print("Asset context rows:   ", len(returns_context))
    print("Asset holdout rows:   ", len(returns_holdout))
    print("Portfolio context rows:", len(port_r_context))
    print("Portfolio holdout rows:", len(port_r_holdout))

    #What actually happened-> Ground truth from 30-day holdout
    realized_vol = port_r_holdout.std(ddof=1)

    # Legacy estimates using only context data
    hist_vol, hist_meta = estimate_volatility(port_r_context, mode="historical")
    rolling_vol, rolling_meta = estimate_volatility(port_r_context, mode="rolling", window=60)
    ewma_vol, ewma_meta = estimate_volatility(port_r_context, mode="ewma")

    print("\n=== Ground Truth ===")
    print("Realized 30-day future portfolio volatility:", realized_vol)

    print("\n=== Legacy Estimates ===")
    print("Historical:", hist_vol)
    print(hist_meta)

    print("\nRolling:")
    print(rolling_vol)
    print(rolling_meta)

    print("\nEWMA:")
    print(ewma_vol)
    print(ewma_meta)

    models = build_models(horizons=[h], tickers=tickers, fetch_start=fetch_start, fetch_end=(forecast_origin.date()),)
    bundle = models[h]
    #Predict
    feature_frames = build_feature_frames(prices_context, returns_context)
    predicted_vols = {}
    for ticker in tickers:
        row = {}
        for feature_name, df in feature_frames.items():
            if ticker not in df.columns:
                raise ValueError(f"{ticker} missing from feature frame {feature_name}")
            row[feature_name] = df.loc[forecast_origin, ticker]
        # one-row DataFrame for this ticker
        X_latest = pd.DataFrame([row])

        # match training column order
        X_latest = X_latest[bundle.feature_columns]

        # scale using trained scaler
        X_latest_scaled = pd.DataFrame(
            bundle.scaler.transform(X_latest),
            columns=X_latest.columns,
        )

        # predict
        pred = bundle.model.predict(X_latest_scaled)[0]
        predicted_vols[ticker] = float(pred)

    print("\n=== Asset-Level Validation ===")
    for ticker in tickers:
        realized_asset_vol_daily = returns_holdout[ticker].std(ddof=1)
        realized_asset_vol_annual = float(realized_asset_vol_daily * np.sqrt(252))

        predicted_asset_vol_annual = predicted_vols[ticker]
        abs_error = abs(predicted_asset_vol_annual - realized_asset_vol_annual)

        print(
            f"{ticker}: "
            f"predicted={predicted_asset_vol_annual:.6f}, "
            f"realized={realized_asset_vol_annual:.6f}, "
            f"abs_error={abs_error:.6f}"
        )

    predicted_vols_daily = {
        ticker: vol / np.sqrt(252)
        for ticker, vol in predicted_vols.items()
    }

    print("\n=== Predicted Asset Vols ===")
    print(predicted_vols_daily)

    cov_matrix = covariance_matrix(returns_context, predicted_vols_daily, weights, corr_method="ewma")
    model_volatility = portfolio_volatility(cov_matrix, weights)
    print(f"\nOur model's prediction of future volatility: {model_volatility}")

if __name__ == "__main__":
    main()