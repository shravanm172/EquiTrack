"""
Rolling walk-forward backtest: GBM vs Heston vs GARCH vs Regime-switching-GMM,
compared head-to-head on distributional forecast quality.

Design:
  - Fixed, EQUAL training window length across all 4 models at every fold
    (controls for training-data quantity, since GMM needs more data than
    GARCH to find stable regimes -- this isolates that variable specifically).
  - Rolling (not expanding) window: at each fold, every model refits fresh
    on exactly TRAIN_WINDOW_DAYS of trailing history, then simulates forward
    HORIZON days. Steps forward by HORIZON each time, same convention as
    rolling_backtest_garch_11.py.
  - Same evaluation metrics for every model, computed once by a shared
    helper so we're not duplicating QLIKE/coverage logic 4 times:
      - forecast variance vs realized variance (RMSE, MAE, QLIKE)
      - 80%/90% interval coverage (is the realized terminal return inside
        the model's own simulated percentile band as often as it should be)
      - 5% VaR exceedance rate

Pipeline:
  1. fit_and_simulate_gbm      -- reuses estimate_drift/estimate_volatility
                                   + simulate_many_paths (already built).
  2. fit_and_simulate_heston   -- reuses calibrate_heston_params +
                                   generate_heston_paths (already built).
  3. fit_and_simulate_garch    -- reuses calibrate_garch_mle +
                                   generate_garch_paths (already built).
  4. fit_and_simulate_regime   -- reuses calibrate_regime_params +
                                   simulate_regime_switching_paths, both from
                                   engines/regime_engine.py (the production
                                   module -- same one forecast_service.py
                                   calls). Same pattern as fit_and_simulate_heston:
                                   the calibration function fetches its own
                                   data given tickers/weights/start/end.
  5. compute_fold_metrics      -- shared scoring, reused for all 4 models.
  6. main                      -- the walk-forward loop + final 4-way
                                   summary comparison table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from providers.market_data import fetch_price_history
from engines.portfolio_engine import (
    prices_to_returns,
    portfolio_value_series,
    portfolio_returns,
)
from engines.forecast_estimators import estimate_drift, estimate_volatility
from engines.stochastic_engine import simulate_many_paths, generate_garch_paths
from engines.garch_engine import calibrate_garch_mle
from engines.heston_engine import calibrate_heston_params, generate_heston_paths
from engines.regime_engine import calibrate_regime_params, simulate_regime_switching_paths

# Config
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
weights = {"AAPL": 0.2, "MSFT": 0.2, "AMZN": 0.2, "GOOGL": 0.2, "NVDA": 0.2}
fetch_start = "2017-03-01"
fetch_end = "2026-03-23"

TRAIN_WINDOW_DAYS = 504   # ~2 years, EQUAL across all 4 models -- the controlled variable
HORIZON = 30              # forecast/simulate 30 trading days ahead
STEP = HORIZON            # non-overlapping folds, same convention as rolling_backtest_garch_11.py
N_PATHS = 1000
N_REGIMES = 3
RANDOM_SEED = 42


def fit_and_simulate_gbm(
    train_returns: pd.Series,
    s0: float,
    horizon: int,
    n_paths: int,
    random_seed: int,
) -> dict:
    """
    Fit constant drift/vol from the training window (mode="mean"/"historical"
    -- the simplest, most literal "GBM with constant drift/vol" baseline),
    then simulate forward with simulate_many_paths.

    Returns: {"prices": (n_paths, horizon+1), "variances": (n_paths, horizon+1)}
    -- same shape/key contract as the GARCH and Heston results, so
    compute_fold_metrics can treat all models identically. GBM has no real
    variance path (constant vol assumption), so "variances" is just
    sigma_daily^2 broadcast across every day/path.
    """
    mu_daily, _ = estimate_drift(train_returns, mode="mean")
    sigma_daily, _ = estimate_volatility(train_returns, mode="historical")

    mu_annual = mu_daily * 252.0
    sigma_annual = sigma_daily * np.sqrt(252.0)
    T = horizon / 252.0

    prices = simulate_many_paths(
        s0=s0, mu=mu_annual, sigma=sigma_annual, T=T, N=horizon, n=n_paths,
        random_seed=random_seed,
    )
    variances = np.full_like(prices, sigma_daily ** 2)

    return {"prices": prices, "variances": variances}


def fit_and_simulate_heston(
    train_start: str,
    train_end: str,
    horizon: int,
    n_paths: int,
    random_seed: int,
) -> dict:
    """
    calibrate_heston_params fetches its own price data internally (given
    tickers/weights/start/end), so this just passes this fold's
    train_start/train_end through, then simulates with generate_heston_paths.
    Note S0 comes from calibrate_heston_params itself (the last portfolio
    value inside [train_start, train_end]) -- no separate s0 argument needed.

    Returns: {"prices": ..., "variances": ...}, same contract as the others.
    """
    params = calibrate_heston_params(
        tickers=tickers, weights=weights, start=train_start, end=train_end,
    )
    T = horizon / 252.0

    sim = generate_heston_paths(
        S0=params.S0, v0=params.v0, mu=params.mu, kappa=params.kappa,
        theta=params.theta, xi=params.xi, rho=params.rho,
        T=T, n_steps=horizon, n_paths=n_paths, random_seed=random_seed,
    )
    # calibrate_heston_params derives theta/v0 from ANNUALIZED variance
    # ((rolling_vol * sqrt(252))**2), so sim.variances is annualized too --
    # convert to daily so it's comparable to GARCH/GBM/Regime's daily-scale
    # variances (otherwise QLIKE/RMSE/MAE are comparing different units).
    variances_daily = sim.variances / 252.0
    return {"prices": sim.prices, "variances": variances_daily}


def _compute_filtered_variances(
    returns: np.ndarray, mu: float, omega: float, alpha: float, beta: float, h0: float,
) -> np.ndarray:
    """
    Same in-sample filtered GARCH(1,1) variance recursion used in
    garch_engine.py / rolling_backtest_garch_11.py / rolling_backtest_garch_residual.py.
    """
    n = len(returns)
    h_path = np.empty(n, dtype=float)
    h_path[0] = max(h0, 1e-12)
    for t in range(1, n):
        eps_prev = returns[t - 1] - mu
        h_path[t] = omega + alpha * (eps_prev ** 2) + beta * h_path[t - 1]
        h_path[t] = max(h_path[t], 1e-12)
    return h_path


def fit_and_simulate_garch(
    train_returns: pd.Series,
    s0: float,
    horizon: int,
    n_paths: int,
    random_seed: int,
) -> dict:
    """
    calibrate_garch_mle(train_returns, estimate_mu=False) + generate_garch_paths.
    Same pattern as rolling_backtest_garch_11.py's main() loop: seed the
    simulation with the LAST FILTERED variance over the training window
    (not the raw params.h0, which is only the MLE optimizer's starting
    guess, not the true end-of-window conditional variance).

    Returns: {"prices": ..., "variances": ...}, same contract as the others.
    """
    r = train_returns.to_numpy(dtype=float)
    params = calibrate_garch_mle(train_returns, estimate_mu=False)

    filtered = _compute_filtered_variances(
        r, params.mu, params.omega, params.alpha, params.beta, params.h0,
    )
    h_last = float(filtered[-1])

    T = horizon / 252.0
    sim = generate_garch_paths(
        S0=s0, mu=params.mu, omega=params.omega, alpha=params.alpha, beta=params.beta,
        h0=h_last, T=T, n_steps=horizon, n_paths=n_paths, nu=params.nu,
        random_seed=random_seed,
    )
    return {"prices": sim.prices, "variances": sim.variances}


def fit_and_simulate_regime(
    train_start: str,
    train_end: str,
    horizon: int,
    n_paths: int,
    random_seed: int,
) -> dict:
    """
    calibrate_regime_params fetches its own price data internally (given
    tickers/weights/start/end), so this just passes this fold's
    train_start/train_end through, then simulates with
    simulate_regime_switching_paths -- calibrate_regime_params is the same
    production calibration function forecast_service.py will call, same
    pattern as fit_and_simulate_heston.
    """
    params = calibrate_regime_params(
        tickers=tickers, weights=weights, start=train_start, end=train_end,
    )

    paths = simulate_regime_switching_paths(
        start_regime_probs=params.current_regime_probs,
        regime_stats=params.regime_stats,
        transition_probs=params.transition_probs,
        s0=params.s0,
        horizon=horizon,
        n_paths=n_paths,
        random_seed=random_seed,
    )

    return paths



def compute_fold_metrics(sim_result: dict, realized_returns: pd.Series) -> dict:
    """
    Shared scoring for all 4 models -- same metrics rolling_backtest_garch_11.py
    already computes for GARCH alone, generalized to work off any model's
    {"prices", "variances"} contract.

    Note the variance window is variances[:, :-1] (columns 0..horizon-1),
    not [:, 1:] like the older script -- per the established convention
    (variances[:, t] is the variance that generated returns[:, t] /
    prices[:, t+1]), that's the precise horizon-length window of variances
    that actually generated the horizon simulated returns.
    """
    prices = sim_result["prices"]
    variances = sim_result["variances"]

    # Simulated (forecast) side
    terminal_returns = prices[:, -1] / prices[:, 0] - 1.0
    forecast_var_paths = np.sum(variances[:, :-1], axis=1)
    forecast_var_mean = float(np.mean(forecast_var_paths))

    p05, p10, p50, p90, p95 = (
        float(x) for x in np.percentile(terminal_returns, [5, 10, 50, 90, 95])
    )

    # Realized (actual) side
    realized = realized_returns.to_numpy(dtype=float)
    realized_terminal_return = float(np.prod(1.0 + realized) - 1.0)
    realized_variance = float(np.sum(np.square(realized)))

    qlike = float(np.log(forecast_var_mean) + realized_variance / forecast_var_mean)

    inside_80 = int(p10 <= realized_terminal_return <= p90)
    inside_90 = int(p05 <= realized_terminal_return <= p95)
    var_5_exceed = int(realized_terminal_return < p05)

    return {
        "forecast_var": forecast_var_mean,
        "realized_var": realized_variance,
        "qlike": qlike,
        "realized_terminal_return": realized_terminal_return,
        "p05": p05, "p10": p10, "p50": p50, "p90": p90, "p95": p95,
        "inside_80": inside_80,
        "inside_90": inside_90,
        "var_5_exceed": var_5_exceed,
    }


def main():
    """
    STEP 6.

    Walk forward through the full date range in TRAIN_WINDOW_DAYS-trailing,
    HORIZON-ahead folds. At each fold, run all 4 fit_and_simulate_* calls on
    IDENTICAL train/test slices, score each with compute_fold_metrics,
    accumulate results, and print a final side-by-side comparison table
    (mean QLIKE, RMSE, coverage, VaR exceedance -- one row per model).
    """

    price_hist = fetch_price_history(tickers=tickers, start=fetch_start, end=fetch_end)
    asset_prices=price_hist.prices
    asset_returns = prices_to_returns(asset_prices)
    port_r = portfolio_returns(asset_returns=asset_returns,weights=weights)

    port_p_df = portfolio_value_series(prices=asset_prices,weights=weights).to_frame(name="portfolio")

    n = len(port_r)
    results = []

    # Rolling, FIXED-length window: [i - TRAIN_WINDOW_DAYS, i) trains, [i, i + HORIZON) tests.
    # Stepping by HORIZON each time keeps folds non-overlapping, same convention as
    # rolling_backtest_garch_11.py.
    for i in range(TRAIN_WINDOW_DAYS, n - HORIZON + 1, STEP):
        train_prices = port_p_df.iloc[i - TRAIN_WINDOW_DAYS : i]
        train_returns = port_r.iloc[i - TRAIN_WINDOW_DAYS : i]   # Series version, for GBM/GARCH
        realized_returns = port_r.iloc[i : i + HORIZON]

        fold_start_date = port_r.index[i]
        fold_end_date = port_r.index[i + HORIZON - 1]
        fold_seed = RANDOM_SEED + i   # vary per fold, same pattern as rolling_backtest_garch_11.py

        s0 = float(train_prices.iloc[-1, 0])
        train_start_date = train_prices.index[0].strftime("%Y-%m-%d")
        train_end_date = train_prices.index[-1].strftime("%Y-%m-%d")

        # All 4 models get IDENTICAL train/test slices for this fold.
        model_outputs = {
            "gbm": fit_and_simulate_gbm(
                train_returns=train_returns, s0=s0, horizon=HORIZON,
                n_paths=N_PATHS, random_seed=fold_seed,
            ),
            "heston": fit_and_simulate_heston(
                train_start=train_start_date, train_end=train_end_date,
                horizon=HORIZON, n_paths=N_PATHS, random_seed=fold_seed,
            ),
            "garch": fit_and_simulate_garch(
                train_returns=train_returns, s0=s0, horizon=HORIZON,
                n_paths=N_PATHS, random_seed=fold_seed,
            ),
            "regime": fit_and_simulate_regime(
                train_start=train_start_date, train_end=train_end_date,
                horizon=HORIZON, n_paths=N_PATHS, random_seed=fold_seed,
            ),
        }

        for model_name, sim_output in model_outputs.items():
            metrics = compute_fold_metrics(sim_result=sim_output, realized_returns=realized_returns)
            metrics["model"] = model_name
            metrics["fold_start"] = fold_start_date
            metrics["fold_end"] = fold_end_date
            results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df["var_error"] = results_df["forecast_var"] - results_df["realized_var"]

    summary = results_df.groupby("model").apply(lambda g: pd.Series({
        "n_folds": len(g),
        "mean_qlike": g["qlike"].mean(),
        "rmse_var": np.sqrt(np.mean(g["var_error"] ** 2)),
        "mae_var": g["var_error"].abs().mean(),
        "coverage_80": g["inside_80"].mean(),
        "coverage_90": g["inside_90"].mean(),
        "var_5_exceed_rate": g["var_5_exceed"].mean(),
    }))

    print("\n=== 4-Model Walk-Forward Comparison ===")
    print(summary)

    return results_df, summary


if __name__ == "__main__":
    main()
