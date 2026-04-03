# For stochastic forecasting
import numpy as np

from dataclasses import asdict, is_dataclass, dataclass

from engines.heston_engine import simulate_baseline_heston_paths, generate_heston_paths, calibrate_heston_params
from engines.garch_engine import calibrate_garch_mle
from engines.forecast_estimators import estimate_volatility, estimate_drift
from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns, portfolio_value_series



def simulate_gbm_path(s0, mu, sigma, T, N, rng):
    '''
        Return path
    '''
    dt = T / N
    path = [s0]
    s = s0
    for _ in range(N):
        z = rng.normal(0,1)
        s = s * np.exp(((mu - 0.5 * sigma**2)*dt) + (sigma * np.sqrt(dt) * z))
        path.append(s)

    return np.array(path)

def simulate_many_paths(s0, mu, sigma, T, N, n, random_seed=None):
    '''
        Return list of paths
    '''
    rng = np.random.default_rng(random_seed)
    paths = []
    for _ in range(n):
        paths.append(simulate_gbm_path(s0, mu, sigma, T, N, rng))
    return np.array(paths)

@dataclass
class GarchSimulationResult:
    prices: np.ndarray
    variances: np.ndarray
    returns: np.ndarray
    params: dict

def generate_garch_paths(
    *,
    S0: float,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    h0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    nu: float,
    random_seed: int | None = None,
) -> GarchSimulationResult:
    """
    Simulate portfolio value paths under GARCH(1,1).

    Assumes mu is per-step drift matching the historical return frequency.
    """
    rng = np.random.default_rng(random_seed)

    prices = np.empty((n_paths,n_steps+1), dtype=float)
    variances = np.empty((n_paths,n_steps+1), dtype=float)
    returns=np.empty((n_paths,n_steps), dtype=float)

    prices[:,0] = S0
    variances[:, 0] = max(h0, 1e-12)

    for t in range(n_steps):
        u = rng.standard_t(df=nu, size=n_paths)
        z = u / (np.sqrt(nu/(nu-2)))
        h_t = np.maximum(variances[:,t], 1e-12)
        r_t = mu + np.sqrt(h_t) * z

        returns[:, t] = r_t
        prices[:, t + 1] = prices[:, t] * (1.0 + r_t)

        eps_t = r_t - mu
        variances[:, t + 1] = omega + alpha * (eps_t ** 2) + beta * h_t
        variances[:, t + 1] = np.maximum(variances[:, t + 1], 1e-12)

    return GarchSimulationResult(
        prices=prices,
        variances=variances,
        returns=returns,
        params={
            "model": "garch",
            "s0": float(S0),
            "mu": float(mu),
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "h0": float(h0),
            "nu": float(nu),
            "T": float(T),
            "N": int(n_steps),
            "n": int(n_paths),
        },
    )


        
def summarize_terminal_metrics(paths):
    """
    Compute terminal-value summary metrics from simulated paths.
    """
    paths = np.asarray(paths)
    terminal_values = paths[:, -1]
    s0 = paths[0, 0]

    median_terminal_value = np.median(terminal_values)
    mean_terminal_value = np.mean(terminal_values)
    bear_case = np.percentile(terminal_values, 10)
    bull_case = np.percentile(terminal_values, 90)
    probability_of_loss = np.mean(terminal_values < s0)

    return {
        "terminal_values": terminal_values,
        "mean_terminal_value": float(mean_terminal_value),
        "median_terminal_value": float(median_terminal_value),
        "bear_case": float(bear_case),
        "bull_case": float(bull_case),
        "probability_of_loss": float(probability_of_loss),
    }


def summarize_drawdown_metrics(paths):
    """
    Compute drawdown-based risk metrics from simulated paths.
    """
    paths = np.asarray(paths)

    running_peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_peaks) / running_peaks
    max_drawdowns = np.min(drawdowns, axis=1)

    median_max_drawdown = np.median(max_drawdowns)
    prob_drawdown_gt_20 = np.mean(max_drawdowns <= -0.20)

    return {
        "max_drawdowns": max_drawdowns,
        "median_max_drawdown": float(median_max_drawdown),
        "prob_drawdown_gt_20": float(prob_drawdown_gt_20),
    }


def summarize_path_metrics(paths):
    """
    Compute percentile path summaries for chart visualization.
    """
    paths = np.asarray(paths)

    p10_path = np.percentile(paths, 10, axis=0)
    p25_path = np.percentile(paths, 25, axis=0)
    p50_path = np.median(paths, axis=0)
    p75_path = np.percentile(paths, 75, axis=0)
    p90_path = np.percentile(paths, 90, axis=0)

    return {
        "p10_path": p10_path,
        "p25_path": p25_path,
        "p50_path": p50_path,
        "p75_path": p75_path,
        "p90_path": p90_path,
    }


def summarize_variance_metrics(variance_paths):
    """
    Compute summary statistics for variance paths when available.
    """
    variance_paths = np.asarray(variance_paths)
    terminal_variances = variance_paths[:, -1]

    return {
        "terminal_variances": terminal_variances,
        "mean_terminal_variance": float(np.mean(terminal_variances)),
        "median_terminal_variance": float(np.median(terminal_variances)),
        "p10_terminal_variance": float(np.percentile(terminal_variances, 10)),
        "p90_terminal_variance": float(np.percentile(terminal_variances, 90)),
    }


def run_stochastic_forecast(*,model, s0, mu=None,  T, N, n, sigma=None,heston_params=None,garch_params=None,random_seed=None):
    '''
        Single entrypoint for stochastic engine
        Supports deterministic constant vol GBM and stochastic vol Heston model
    '''
    model = model.lower().strip()

    if model == "gbm":
        if sigma is None:
            raise ValueError("sigma is required when model=gbm")
        if mu is None:
            raise ValueError("mu is required when model='gbm'")

        price_paths = simulate_many_paths(s0=s0, mu=mu, sigma=sigma, T=T, N=N, n=n,random_seed=random_seed)

        variance_paths = None

        model_params = {
            "model": "gbm",
            "s0": float(s0),
            "mu": float(mu),
            "sigma": float(sigma),
            "T": float(T),
            "N": int(N),
            "n": int(n),
        }
    
    elif model == "heston":
        if mu is None:
            raise ValueError("mu is required when model='heston'")

        if heston_params is None:
            raise ValueError("heston_params is required when model='heston'.")
        if is_dataclass(heston_params):
            hp = asdict(heston_params)
        elif isinstance(heston_params, dict):
            hp = heston_params
        else:
            raise TypeError("heston_params must be a dict or HestonCalibratedParams dataclass.")
        
        result = generate_heston_paths(
            S0=s0,
            v0=hp["v0"],
            mu=mu,
            kappa=hp["kappa"],
            theta=hp["theta"],
            xi=hp["xi"],
            rho=hp["rho"],
            T=T,
            n_steps=N,
            n_paths=n,
            random_seed=random_seed,
        )

        price_paths = result.prices
        variance_paths = result.variances

        if hasattr(result, "params") and result.params is not None:
            if is_dataclass(result.params):
                model_params = asdict(result.params)
            else:
                model_params = result.params
        else:
            model_params = {
                "model": "heston",
                "s0": float(s0),
                "mu": float(mu),
                "v0": float(hp["v0"]),
                "kappa": float(hp["kappa"]),
                "theta": float(hp["theta"]),
                "xi": float(hp["xi"]),
                "rho": float(hp["rho"]),
                "T": float(T),
                "N": int(N),
                "n": int(n),
            }
    
    elif model == "garch":
        if garch_params is None:
            raise ValueError("garch_params is required when model is 'garch'.")
        
        if is_dataclass(garch_params):
            gp = asdict(garch_params)
        elif isinstance(garch_params, dict):
            gp = garch_params
        else:
            raise TypeError("garch_params must be a dict or GarchCalibratedParams dataclass.")
        
        garch_mu = gp.get("mu", mu)
        if garch_mu is None:
            raise ValueError("mu must be provided either in garch_params or as the top-level mu argument.")

        result = generate_garch_paths(
            S0=s0,
            mu=garch_mu,
            omega=gp["omega"],
            alpha=gp["alpha"],
            beta=gp["beta"],
            h0=gp["h0"],
            T=T,
            n_steps=N,
            n_paths=n,
            nu=gp["nu"],
            random_seed=random_seed,
        )

        price_paths = result.prices
        variance_paths = result.variances
        model_params = result.params

    else:
        raise ValueError("model must be either 'gbm', 'heston', or 'garch'.")

    terminal = summarize_terminal_metrics(price_paths)
    drawdown = summarize_drawdown_metrics(price_paths)
    path_metrics = summarize_path_metrics(price_paths)

    output = {
        "model": model,
        "paths": price_paths,
        "variance_paths": variance_paths,
        "terminal": terminal,
        "drawdown": drawdown,
        "path_metrics": path_metrics,
        "params": model_params,
    }

    if variance_paths is not None:
        output["variance"] = summarize_variance_metrics(variance_paths)

    return output

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