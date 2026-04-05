"""
Rolling backtest comparison: our GJR-GARCH vs arch package reference.
Same data, same windows, same metrics. Runs on multiple portfolios.
"""
from __future__ import annotations
from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns, portfolio_value_series
from engines.garch_engine import calibrate_garch_mle, gjr_future_horizon_variance
from engines.stochastic_engine import generate_garch_paths
from arch import arch_model
import numpy as np
import pandas as pd

# ── Settings ──────────────────────────────────────────────
HORIZON       = 20
TRAIN_WINDOW  = 504
STEP          = HORIZON
N_PATHS       = 1_000
RANDOM_SEED   = 42
FETCH_START   = "2017-01-01"
FETCH_END     = "2026-01-01"

PORTFOLIOS = {
    "Tech (high-beta)": {
        "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "weights": {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "NVDA": 0.25},
    },
    "Defensive (low-beta, diversified)": {
        "tickers": ["JNJ", "PG", "XOM", "JPM"],
        "weights": {"JNJ": 0.25, "PG": 0.25, "XOM": 0.25, "JPM": 0.25},
    },
}


def _filtered_variances(returns, omega, alpha, beta, lam, h0):
    n = len(returns)
    h = np.empty(n)
    h[0] = max(h0, 1e-12)
    for t in range(1, n):
        eps = returns[t - 1]
        ind = 1.0 if eps < 0.0 else 0.0
        h[t] = omega + alpha * eps**2 + lam * ind * eps**2 + beta * h[t - 1]
        h[t] = max(h[t], 1e-12)
    return h


def _run_fold_ours(train_returns, future_returns, S0, seed_offset):
    params = calibrate_garch_mle(returns=train_returns, estimate_mu=False, model="gjr")

    fv = _filtered_variances(
        train_returns.to_numpy(float),
        params.omega, params.alpha, params.beta, params.lam, params.h0,
    )
    h_last = float(fv[-1])

    forecast_var = gjr_future_horizon_variance(
        omega=params.omega, alpha=params.alpha, beta=params.beta,
        lam=params.lam, h_t=h_last, horizon=HORIZON,
    )

    sim = generate_garch_paths(
        S0=S0, mu=params.mu, omega=params.omega, alpha=params.alpha,
        beta=params.beta, lam=params.lam, h0=h_last,
        T=HORIZON / 252.0, n_steps=HORIZON, n_paths=N_PATHS,
        nu=params.nu, random_seed=RANDOM_SEED + seed_offset,
    )
    term_ret = (sim.prices[:, -1] / sim.prices[:, 0]) - 1.0
    return params, forecast_var, term_ret


def _run_fold_arch(train_returns, future_returns, S0, seed_offset):
    # arch works in percentage returns
    y = train_returns * 100
    am = arch_model(y, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="StudentsT")
    res = am.fit(disp="off", show_warning=False)

    # Extract params (scale omega back to decimal)
    omega = res.params["omega"] / 1e4
    alpha = res.params["alpha[1]"]
    gamma = res.params["gamma[1]"]   # = lambda
    beta  = res.params["beta[1]"]
    nu    = res.params["nu"]

    # Filtered variances from arch (in %^2), convert to decimal
    cond_var = res.conditional_volatility**2 / 1e4
    h_last = float(cond_var.iloc[-1])

    # Analytical forecast variance (same formula)
    forecast_var = gjr_future_horizon_variance(
        omega=omega, alpha=alpha, beta=beta,
        lam=gamma, h_t=h_last, horizon=HORIZON,
    )

    # Simulate with same engine for apples-to-apples path generation
    sim = generate_garch_paths(
        S0=S0, mu=0.0, omega=omega, alpha=alpha,
        beta=beta, lam=gamma, h0=h_last,
        T=HORIZON / 252.0, n_steps=HORIZON, n_paths=N_PATHS,
        nu=nu, random_seed=RANDOM_SEED + seed_offset,
    )
    term_ret = (sim.prices[:, -1] / sim.prices[:, 0]) - 1.0

    class _P:
        pass
    params = _P()
    params.omega = omega; params.alpha = alpha; params.beta = beta
    params.lam = gamma; params.nu = nu; params.mu = 0.0
    return params, forecast_var, term_ret


def _score_fold(forecast_var, term_ret, future_returns):
    realized_var = float(np.sum(np.square(future_returns.to_numpy(float))))
    realized_ret = float(np.prod(1.0 + future_returns.to_numpy(float)) - 1.0)

    qlike = float(np.log(forecast_var) + realized_var / forecast_var - 1.0)

    p05 = float(np.percentile(term_ret, 5))
    p10 = float(np.percentile(term_ret, 10))
    p90 = float(np.percentile(term_ret, 90))
    p95 = float(np.percentile(term_ret, 95))

    return {
        "forecast_var": forecast_var,
        "realized_var": realized_var,
        "qlike": qlike,
        "inside_80": int(p10 <= realized_ret <= p90),
        "inside_90": int(p05 <= realized_ret <= p95),
        "var_5_exceed": int(realized_ret < p05),
    }


def main():
    for port_name, port_cfg in PORTFOLIOS.items():
        tickers = port_cfg["tickers"]
        weights = port_cfg["weights"]

        print(f"\n{'#' * 70}")
        print(f"  PORTFOLIO: {port_name}")
        print(f"  Tickers: {tickers}   Weights: equal 25%")
        print(f"{'#' * 70}")

        ph = fetch_price_history(tickers=tickers, start=FETCH_START, end=FETCH_END)
        ar = prices_to_returns(ph.prices)
        port_p = portfolio_value_series(ph.prices, weights).dropna()
        port_r = portfolio_returns(asset_returns=ar, weights=weights).dropna()
        idx = port_p.index.intersection(port_r.index)
        port_p, port_r = port_p.loc[idx], port_r.loc[idx]

        ours_rows, arch_rows = [], []
        ours_params_list, arch_params_list = [], []
        n_folds = 0

        for fi in range(TRAIN_WINDOW, len(port_r) - HORIZON + 1, STEP):
            train_ret = port_r.iloc[fi - TRAIN_WINDOW : fi]
            future_ret = port_r.iloc[fi : fi + HORIZON]
            S0 = float(port_p.iloc[fi])

            try:
                p_o, fv_o, tr_o = _run_fold_ours(train_ret, future_ret, S0, fi)
                ours_rows.append(_score_fold(fv_o, tr_o, future_ret))
                ours_params_list.append({
                    "alpha": p_o.alpha, "beta": p_o.beta,
                    "lam": p_o.lam, "nu": p_o.nu,
                })
            except Exception as e:
                print(f"[Ours ] fold {n_folds} failed: {e}")
                continue

            try:
                p_a, fv_a, tr_a = _run_fold_arch(train_ret, future_ret, S0, fi)
                arch_rows.append(_score_fold(fv_a, tr_a, future_ret))
                arch_params_list.append({
                    "alpha": p_a.alpha, "beta": p_a.beta,
                    "lam": p_a.lam, "nu": p_a.nu,
                })
            except Exception as e:
                print(f"[arch ] fold {n_folds} failed: {e}")
                ours_rows.pop()
                ours_params_list.pop()
                continue

            n_folds += 1

        df_o = pd.DataFrame(ours_rows)
        df_a = pd.DataFrame(arch_rows)
        df_op = pd.DataFrame(ours_params_list)
        df_ap = pd.DataFrame(arch_params_list)

        def _summary(df, dfp, label):
            print(f"\n{'=' * 60}")
            print(f"  {label}   ({len(df)} folds)")
            print(f"{'=' * 60}")
            print(f"  Mean QLIKE:             {df['qlike'].mean():.6f}")
            rmse = np.sqrt(np.mean((df["forecast_var"] - df["realized_var"])**2))
            mae  = np.mean(np.abs(df["forecast_var"] - df["realized_var"]))
            print(f"  RMSE variance:          {rmse:.6f}")
            print(f"  MAE variance:           {mae:.6f}")
            print(f"  80% interval coverage:  {df['inside_80'].mean():.2%}")
            print(f"  90% interval coverage:  {df['inside_90'].mean():.2%}")
            print(f"  5% VaR exceedance rate: {df['var_5_exceed'].mean():.2%}")
            print(f"  --- Mean params ---")
            print(f"  Mean alpha:  {dfp['alpha'].mean():.6e}")
            print(f"  Mean beta:   {dfp['beta'].mean():.6f}")
            print(f"  Mean lambda: {dfp['lam'].mean():.6f}")
            print(f"  Mean nu:     {dfp['nu'].mean():.4f}")

        _summary(df_o, df_op, f"Our GJR-GARCH — {port_name}")
        _summary(df_a, df_ap, f"arch GJR-GARCH — {port_name}")


if __name__ == "__main__":
    main()
