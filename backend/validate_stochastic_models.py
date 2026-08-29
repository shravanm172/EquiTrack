"""
Standalone validation/demo script for the stochastic forecast engine
(GBM/Heston/GARCH/Regime). Moved out of engines/stochastic_engine.py, which
should only contain production entrypoints -- this is demo/print/plot
tooling, same role as validate_garch.py.
"""
from __future__ import annotations

import numpy as np

from engines.stochastic_engine import run_stochastic_forecast
from engines.garch_engine import calibrate_garch_mle
from engines.forecast_estimators import estimate_drift, estimate_volatility
from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns, portfolio_value_series


def print_forecast_summary(label, forecast):
    """
    Pretty-print summary metrics for one forecast run.
    """
    terminal = forecast["terminal"]
    drawdown = forecast["drawdown"]

    print(f"\n=== {label} ===")
    print(f"Model: {forecast['model']}")
    print(f"Mean terminal value:   {terminal['mean_terminal_value']:.4f}")
    print(f"Median terminal value: {terminal['median_terminal_value']:.4f}")
    print(f"Bear case (10th %):    {terminal['bear_case']:.4f}")
    print(f"Bull case (90th %):    {terminal['bull_case']:.4f}")
    print(f"Probability of loss:   {terminal['probability_of_loss']:.2%}")
    print(f"Median max drawdown:   {drawdown['median_max_drawdown']:.2%}")
    print(f"P(drawdown <= -20%):   {drawdown['prob_drawdown_gt_20']:.2%}")

    if "variance" in forecast:
        variance = forecast["variance"]
        print("Terminal variance summary:")
        print(f"  Mean terminal variance:   {variance['mean_terminal_variance']:.6f}")
        print(f"  Median terminal variance: {variance['median_terminal_variance']:.6f}")
        print(f"  10th percentile variance: {variance['p10_terminal_variance']:.6f}")
        print(f"  90th percentile variance: {variance['p90_terminal_variance']:.6f}")


def print_comparison(gbm_forecast, heston_forecast):
    """
    Print a direct side-by-side comparison of key summary metrics.
    """
    gbm_t = gbm_forecast["terminal"]
    gbm_d = gbm_forecast["drawdown"]

    hes_t = heston_forecast["terminal"]
    hes_d = heston_forecast["drawdown"]

    print("\n=== Side-by-Side Comparison ===")
    print(f"{'Metric':30s} {'GBM':>15s} {'Heston':>15s}")
    print("-" * 62)
    print(f"{'Mean terminal value':30s} {gbm_t['mean_terminal_value']:15.4f} {hes_t['mean_terminal_value']:15.4f}")
    print(f"{'Median terminal value':30s} {gbm_t['median_terminal_value']:15.4f} {hes_t['median_terminal_value']:15.4f}")
    print(f"{'Bear case (10th %)':30s} {gbm_t['bear_case']:15.4f} {hes_t['bear_case']:15.4f}")
    print(f"{'Bull case (90th %)':30s} {gbm_t['bull_case']:15.4f} {hes_t['bull_case']:15.4f}")
    print(f"{'Probability of loss':30s} {gbm_t['probability_of_loss']:15.2%} {hes_t['probability_of_loss']:15.2%}")
    print(f"{'Median max drawdown':30s} {gbm_d['median_max_drawdown']:15.2%} {hes_d['median_max_drawdown']:15.2%}")
    print(f"{'P(drawdown <= -20%)':30s} {gbm_d['prob_drawdown_gt_20']:15.2%} {hes_d['prob_drawdown_gt_20']:15.2%}")


def plot_comparison(garch_forecast, heston_forecast, T, N):
    """
    Plot percentile fan charts for garch and Heston.
    """
    import matplotlib.pyplot as plt

    x = np.linspace(0, T, N + 1)

    garch_paths = garch_forecast["path_metrics"]
    hes_paths = heston_forecast["path_metrics"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, garch_paths["p50_path"], label="GBM median path")
    plt.fill_between(x, garch_paths["p25_path"], garch_paths["p75_path"], alpha=0.3, label="GBM 25-75%")
    plt.fill_between(x, garch_paths["p10_path"], garch_paths["p90_path"], alpha=0.2, label="GBM 10-90%")
    plt.title("GARCH Monte Carlo Forecast")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, hes_paths["p50_path"], label="Heston median path")
    plt.fill_between(x, hes_paths["p25_path"], hes_paths["p75_path"], alpha=0.3, label="Heston 25-75%")
    plt.fill_between(x, hes_paths["p10_path"], hes_paths["p90_path"], alpha=0.2, label="Heston 10-90%")
    plt.title("Heston Monte Carlo Forecast")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, garch_paths["p50_path"], label="GARCH median path")
    plt.plot(x, hes_paths["p50_path"], label="Heston median path")
    plt.fill_between(x, garch_paths["p10_path"], garch_paths["p90_path"], alpha=0.15, label="GBM 10-90%")
    plt.fill_between(x, hes_paths["p10_path"], hes_paths["p90_path"], alpha=0.15, label="Heston 10-90%")
    plt.title("GARCH vs Heston Median / Tail Comparison")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()


def main():
    """
    Compare deterministic-vol GBM vs stochastic-vol Heston
    using the same initial portfolio setup.
    """
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
    prices = price_history.prices
    asset_returns = prices_to_returns(prices)
    port_r = portfolio_returns(asset_returns=asset_returns, weights=weights)

    # Shared forecast inputs
    port_p = portfolio_value_series(prices,weights)
    s0 = float(port_p.iloc[-1])


    mu_daily, drift_meta = estimate_drift(port_r, "ewma", lam=0.94)
    mu = float(mu_daily * 252)

    sigma_daily, vol_meta = estimate_volatility(port_r=port_r, mode="ewma", lam=0.94)  # example deterministic vol input
    sigma = float(sigma_daily * np.sqrt(252))

    T = 1.0                 # 1 year
    N = 252                 # trading days
    n = 200                # number of Monte Carlo paths to generate
    random_seed = 6996

    # Fair comparison:
    # Heston initial variance should correspond to GBM volatility squared
    # heston_params = calibrate_heston_params(tickers=tickers, weights=weights, start=fetch_start, end=fetch_end, window=21, annualization_factor=252, portfolio_name="big_tech")

    #Remember to use mu_daily if estimate_mu is false
    garch_params = calibrate_garch_mle(returns=port_r, estimate_mu=False)

    print(garch_params)

    long_run_var = garch_params.omega / (1.0 - garch_params.alpha - garch_params.beta)
    print("Long-run variance:", long_run_var)
    print("Long-run daily vol:", np.sqrt(long_run_var))
    print("Long-run annual vol:", np.sqrt(long_run_var) * np.sqrt(252))

    print("Sample daily variance:", np.var(port_r, ddof=1))
    print("Sample daily vol:", np.std(port_r, ddof=1))
    print("Sample annual vol:", np.std(port_r, ddof=1) * np.sqrt(252))

    # gbm_forecast = run_stochastic_forecast(
    #     model="gbm",
    #     s0=s0,
    #     mu=mu,
    #     sigma=sigma,
    #     T=T,
    #     N=N,
    #     n=n,
    #     random_seed=random_seed,
    # )

    # heston_forecast = run_stochastic_forecast(
    #     model="heston",
    #     s0=s0,
    #     mu=mu,
    #     T=T,
    #     N=N,
    #     n=n,
    #     heston_params=heston_params,
    #     random_seed=random_seed,
    # )

    garch_forecast = run_stochastic_forecast(
        model="garch",
        s0=s0,
        T=T,
        N=N,
        n=n,
        garch_params=garch_params,
        random_seed=random_seed
    )

    print_forecast_summary("GARCH Forecast Summary", garch_forecast)
    # print_forecast_summary("GBM Forecast Summary", gbm_forecast)
    # print_forecast_summary("Heston Forecast Summary", heston_forecast)
    # print_comparison(garch_forecast, heston_forecast)

    # plot_comparison(garch_forecast, heston_forecast, T=T, N=N)


if __name__=="__main__":
    main()
