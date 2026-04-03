from __future__ import annotations
from engines.stochastic_engine import generate_garch_paths, GarchSimulationResult
from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns, portfolio_value_series
from engines.garch_engine import calibrate_garch_mle

import numpy as np
import pandas as pd


def _compute_filtered_garch_variances(
    returns: np.ndarray,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    h0: float,
) -> np.ndarray:
    """
    Compute in-sample filtered conditional variances using observed returns.
    """
    n = len(returns)
    h = np.empty(n, dtype=float)
    h[0] = max(h0, 1e-12)

    for t in range(1, n):
        eps_prev = returns[t - 1] - mu
        h[t] = omega + alpha * (eps_prev ** 2) + beta * h[t - 1]
        h[t] = max(h[t], 1e-12)

    return h

def main():
    horizon = 20
    train_window=252 #rolling window
    step=horizon
    n_paths=1000 #number of paths to simulate at each step
    random_seed=42

    fetch_start = "2017-01-01"
    fetch_end = "2026-01-01"
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    weights = {
        "AAPL": 0.25,
        "MSFT":0.25,
        "GOOGL":0.25,
        "NVDA":0.25,
    }

    price_history = fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    asset_prices = price_history.prices
    asset_returns = prices_to_returns(asset_prices)
    port_p = portfolio_value_series(asset_prices, weights).dropna()
    port_r = portfolio_returns(asset_returns=asset_returns, weights=weights).dropna()

    # align just in case
    common_index = port_p.index.intersection(port_r.index)
    port_p = port_p.loc[common_index]
    port_r = port_r.loc[common_index]

    if len(port_r) < train_window + horizon:
        raise ValueError("Not enough data for rolling backtest.")
    
    results=[]

    for forecast_start_idx in range(train_window, len(port_r) - horizon + 1, step):
        train_returns = port_r.iloc[forecast_start_idx - train_window : forecast_start_idx]
        future_returns = port_r.iloc[forecast_start_idx : forecast_start_idx + horizon]

        forecast_start_date = port_r.index[forecast_start_idx]
        forecast_end_date = port_r.index[forecast_start_idx + horizon - 1]

        # use the portfolio value at forecast origin
        S0 = float(port_p.iloc[forecast_start_idx])

        # 1. fit GARCH on training window
        try:
            params = calibrate_garch_mle(
                returns=train_returns,
                estimate_mu=False,
            )
        except Exception as e:
            print("Calibration failed")
            print("forecast_start_idx:", forecast_start_idx)
            print("forecast_start_date:", forecast_start_date)
            print("train mean:", float(train_returns.mean()))
            print("train var:", float(train_returns.var(ddof=1)))
            print("train min:", float(train_returns.min()))
            print("train max:", float(train_returns.max()))
            raise

        # 2. compute filtered variance path on training data
        filtered_variances = _compute_filtered_garch_variances(
            returns=train_returns.to_numpy(dtype=float),
            mu=params.mu,
            omega=params.omega,
            alpha=params.alpha,
            beta=params.beta,
            h0=params.h0,
        )

        # 3. last filtered variance becomes forecast initial variance
        h_last = float(filtered_variances[-1])

        # 4. simulate forward
        sim = generate_garch_paths(
            S0=S0,
            mu=params.mu,
            omega=params.omega,
            alpha=params.alpha,
            beta=params.beta,
            h0=h_last,
            T=horizon / 252.0,   # metadata only if you still keep T
            n_steps=horizon,
            n_paths=n_paths,
            nu=params.nu,
            random_seed=random_seed+forecast_start_idx,
        )

        # -----------------------------
        # Forecast objects from simulation
        # -----------------------------

        # terminal simulated arithmetic returns
        terminal_returns = (sim.prices[:, -1] / sim.prices[:, 0]) - 1.0

        # use the variances that generated each simulated return
        forecast_var_paths = np.sum(sim.variances[:, 1:], axis=1)
        forecast_var_mean = float(np.mean(forecast_var_paths))

        # interval / VaR estimates
        p05 = float(np.percentile(terminal_returns, 5))
        p10 = float(np.percentile(terminal_returns, 10))
        p50 = float(np.percentile(terminal_returns, 50))
        p90 = float(np.percentile(terminal_returns, 90))
        p95 = float(np.percentile(terminal_returns, 95))

        # -----------------------------
        # Realized outcomes
        # -----------------------------

        realized_terminal_return = float(np.prod(1.0 + future_returns.to_numpy(dtype=float)) - 1.0)
        realized_variance = float(np.sum(np.square(future_returns.to_numpy(dtype=float))))

        # QLIKE
        qlike = float(np.log(forecast_var_mean) + realized_variance / forecast_var_mean)

        # coverage flags
        inside_80 = int(p10 <= realized_terminal_return <= p90)
        inside_90 = int(p05 <= realized_terminal_return <= p95)

        # 5% VaR exceedance
        var_5_exceed = int(realized_terminal_return < p05)

        results.append({
            "forecast_start": forecast_start_date,
            "forecast_end": forecast_end_date,
            "S0": S0,

            "mu": float(params.mu),
            "omega": float(params.omega),
            "alpha": float(params.alpha),
            "beta": float(params.beta),
            "nu": float(params.nu),
            "h_last": h_last,

            "forecast_var": forecast_var_mean,
            "realized_var": realized_variance,
            "qlike": qlike,

            "realized_terminal_return": realized_terminal_return,

            "p05": p05,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "p95": p95,

            "inside_80": inside_80,
            "inside_90": inside_90,
            "var_5_exceed": var_5_exceed,
        })

    backtest_df = pd.DataFrame(results)

    
    print(backtest_df[['alpha','beta','nu']].tail(10))

    print("\n=== GARCH Rolling Backtest Summary ===")
    print(f"Number of folds: {len(backtest_df)}")
    print(f"Mean QLIKE: {backtest_df['qlike'].mean():.6f}")
    print(f"RMSE variance: {np.sqrt(np.mean((backtest_df['forecast_var'] - backtest_df['realized_var'])**2)):.6f}")
    print(f"MAE variance: {np.mean(np.abs(backtest_df['forecast_var'] - backtest_df['realized_var'])):.6f}")
    print(f"80% interval coverage: {backtest_df['inside_80'].mean():.2%}")
    print(f"90% interval coverage: {backtest_df['inside_90'].mean():.2%}")
    print(f"5% VaR exceedance rate: {backtest_df['var_5_exceed'].mean():.2%}")

    print("\nLast few folds:")
    print(backtest_df.tail())

if __name__ == "__main__":
    main()