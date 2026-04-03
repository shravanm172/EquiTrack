from __future__ import annotations

import pandas as pd
import numpy as np

from providers.market_data import fetch_price_history
from engines.portfolio_engine import portfolio_value_series, portfolio_log_returns_from_value
from engines.forecast_estimators import estimate_drift, estimate_volatility
from engines.stochastic_engine import run_stochastic_forecast

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, brier_score_loss

from engines.features.volatility_features import rolling_volatility, ewm_volatility, vol_change, vol_of_vol, vol_ratio, vol_short_long_ratio
from engines.features.drawdown_features import drawdown
from engines.features.absolute_returns_features import ewma_abs_returns, rolling_abs_returns
from engines.features.moving_average_features import moving_average_ratio
from engines.features.momentum_features import momentum_swings, momentum

from engines.ml.dataset_builder import assemble_dataset

def build_feature_frames(prices: pd.DataFrame, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build the feature frames used by the probability-of-loss model.
    """
    rolling_vol_20_df = rolling_volatility(returns, window=20)
    ewm_vol_df = ewm_volatility(returns)
    vol_of_vol_df = vol_of_vol(ewm_vol_df, window=20)
    vol_change_df = vol_change(ewm_vol_df)
    vol_ratio_lag_5_df = vol_ratio(ewm_vol_df, lag=5)
    vol_short_long_ratio_df = vol_short_long_ratio(returns)

    ewma_abs_df = ewma_abs_returns(returns)
    rolling_abs_20_df = rolling_abs_returns(returns, window=20)

    momentum_10_df = momentum(returns, window=10)
    momentum_swings_10_df = momentum_swings(returns, window=10)

    ma_ratio_df = moving_average_ratio(prices, short_window=20, long_window=100).reindex(returns.index)
    drawdown_df = drawdown(prices).reindex(returns.index)

    features = {
        "rolling_vol_20": rolling_vol_20_df,
        "ewm_volatility": ewm_vol_df,
        "vol_of_vol_20": vol_of_vol_df,
        "vol_change": vol_change_df,
        "vol_ratio_lag_5": vol_ratio_lag_5_df,
        "vol_short_long_ratio": vol_short_long_ratio_df,
        "ewma_abs_returns": ewma_abs_df,
        "rolling_abs_returns_20": rolling_abs_20_df,
        "momentum_10": momentum_10_df,
        "momentum_swings_10": momentum_swings_10_df,
        "moving_avg_ratio_20_100": ma_ratio_df,
        "drawdown": drawdown_df,
    }

    return features

def build_target_df(returns: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Build binary target: 1 if loss over next h steps, else 0.
    Uses log returns.
    """
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if h <= 0:
        raise ValueError("h must be positive.")

    future_log_return = returns.rolling(window=h).sum().shift(-h + 1)
    target_df = (future_log_return < 0).astype(int)
    target_df.columns = ["target"]
    return target_df


def monte_carlo_prob_of_loss(portfolio_values:pd.Series, portfolio_returns:pd.Series, current_date:pd.Timestamp, horizon:int, n_paths:int=1000, random_seed:int=42) -> float:
    """
    Run monte carlo simulations to forecast prob_of_loss at each time t as a side-by-side comparison with the ML model's predicted probability of loss
    Used as a benchmark for evaluating the model's performance
    Currently, the monte carlo model uses the heston model to simulate volatilty, with historically calibrated baseline parameters
    (Well actually we'll just use the standard GBM Monte Carlo simulation first)
    """
    hist_values = portfolio_values.loc[:current_date]
    hist_returns = portfolio_returns.loc[:current_date]

    #Compute parameters for simulation
    s0 = hist_values.iloc[-1]
    mu_daily, drift_meta = estimate_drift(port_r=hist_returns, mode="ewma", lam=0.94)
    mu = float(mu_daily * 252)
    sigma_daily, vol_meta = estimate_volatility(port_r=hist_returns, mode="ewma", lam=0.94)  # example deterministic vol input
    sigma = float(sigma_daily * np.sqrt(252))

    N=horizon
    T=horizon / 252

    res = run_stochastic_forecast(
        model="gbm",
        s0=float(s0),
        mu=mu,
        sigma=sigma,
        T=T,
        N=N,
        n=n_paths,
        random_seed=random_seed,
    )
   
    return float(res["terminal"]["probability_of_loss"])


def main():
    tickers = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL"]
    weights = {
        "AAPL":0.2,
        "NVDA":0.2,
        "MSFT":0.2,
        "AMZN":0.2,
        "GOOGL":0.2,
    }
    
    fetch_start = "2017-01-01"
    fetch_end = "2024-12-31"
    #2025-present was deliberately held out as the test set

    # horizon = 20
    horizons = [20]
    
    price_history=fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    prices=price_history.prices
    port_p = portfolio_value_series(prices=prices, weights=weights)
    port_r = portfolio_log_returns_from_value(port_p)

    port_p_df = port_p.to_frame()
    port_r_df = port_r.to_frame()

    features_dict = build_feature_frames(prices=port_p_df, returns=port_r_df)
    

    # for name, df in features_dict.items():
    #     print(name, df.shape, df.isna().sum().sum())
    #     print(name, df.index.min(), df.index.max(), df.index.equals(port_r_df.index))
    #     print(end="\n")

    # print("port_p_df shape:", port_p_df.shape)
    # print("port_r_df shape:", port_r_df.shape)
    # print("target_df shape:", target_df.shape)
    # print("X_full shape:", X_full.shape)
    # print("y_full shape:", y_full.shape)

    summary_results = []
    #Loop over all horizons
    for h in horizons:
        print(f"\n=== Horizon: {h} ===")
        target_df = build_target_df(port_r_df, h=h)
        X_full, y_full = assemble_dataset(features=features_dict,target_df=target_df)
        
        decision_threshold = 0.4

        results = []
        #Rolling validation loop: Train on [0:t), predict at t.
        min_train_size = 252
        eval_start = pd.Timestamp("2020-01-01")
        for i in range(min_train_size,len(X_full)):
            
            current_date = X_full.index[i]
            if current_date < eval_start:
                continue

            X_train = X_full.iloc[:i]
            y_train = y_full.iloc[:i]
            X_val = X_full.iloc[[i]]
            y_val = y_full.iloc[i]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # model = LogisticRegression(max_iter=1000, C=10)
            model = XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.7,
                reg_lambda=1.0,
                reg_alpha=0.1,
                min_child_weight=5,
                gamma=1.0,
                random_state=42,
                eval_metric="logloss",
            )

            model.fit(X_train_scaled, y_train)

            #Track training errors as well
            train_prob = model.predict_proba(X_train_scaled)[:, 1]
            train_pred = (train_prob > decision_threshold).astype(int)
        
            y_prob = model.predict_proba(X_val_scaled)[0, 1]
            y_pred = int(y_prob > decision_threshold)# Altering decision threshold to balance precision/recall tradeoff

            #Run monte carlo simulations at t side-by-side to benchmark performance
            # mc_prob = monte_carlo_prob_of_loss(
            #     portfolio_values=port_p,
            #     portfolio_returns=port_r,
            #     current_date=current_date,
            #     horizon=h,
            #     n_paths=200,
            #     random_seed=42,
            # )
            

            results.append({
                "date": current_date,
                "y_true": y_val,
                "y_pred": y_pred,
                "y_prob": y_prob,
                # "mc_prob":mc_prob,
                "train_auc": roc_auc_score(y_train, train_prob),
                "train_brier": brier_score_loss(y_train, train_prob),
                "train_acc": accuracy_score(y_train, train_pred),
            })

        if not results:
            print(f"No results for horizon {h}")
            continue

        results_df = pd.DataFrame(results).set_index("date")
        y_true = results_df["y_true"]
        y_pred = results_df["y_pred"]
        ml_prob = results_df["y_prob"]
        # mc_prob = results_df["mc_prob"]


        ml_auc = roc_auc_score(y_true, ml_prob)
        # mc_auc = roc_auc_score(y_true, mc_prob)
        ml_brier = brier_score_loss(y_true, ml_prob)
        # mc_brier = brier_score_loss(y_true, mc_prob)

        avg_train_auc = results_df["train_auc"].mean()
        avg_train_brier = results_df["train_brier"].mean()
        avg_train_acc = results_df["train_acc"].mean()

        summary_results.append({
            "horizon": h,
            "ml_auc": ml_auc,
            # "mc_auc": mc_auc,
            "ml_brier": ml_brier,
            # "mc_brier": mc_brier,
        })

        print("Base rate:", y_true.mean())

        print("ML Accuracy:", accuracy_score(y_true, y_pred))
        print("ML Precision:", precision_score(y_true, y_pred))
        print("ML Recall:", recall_score(y_true, y_pred))
        print("ML ROC AUC:", ml_auc)
        # print("MC ROC AUC:", mc_auc)

        print("ML Brier:", ml_brier)
        # print("MC Brier:", mc_brier)

        print("Avg Train Accuracy:", avg_train_acc)
        print("Avg Train AUC:", avg_train_auc)
        print("Avg Train Brier:", avg_train_brier)

    summary_df = pd.DataFrame(summary_results).set_index("horizon")
    print("\n=== Summary ===")
    print(summary_df)

if __name__ == "__main__":
    main()



