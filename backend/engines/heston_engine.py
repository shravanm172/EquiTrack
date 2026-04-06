from __future__ import annotations

from providers.market_data import fetch_price_history
from engines.portfolio_engine import portfolio_value_series, portfolio_log_returns_from_value

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class HestonSimulationResult:
    times: np.ndarray         
    prices: np.ndarray        
    variances: np.ndarray 
    params: HestonCalibratedParams | None = None

@dataclass(frozen=True)
class HestonCalibratedParams:
    portfolio_name: str
    start: str
    end: str
    window: int

    S0: float
    mu: float
    v0: float
    theta: float
    kappa: float
    xi: float
    rho: float

    realized_vol: pd.Series
    realized_var: pd.Series
    portfolio_returns: pd.Series
    portfolio_prices: pd.Series

def generate_heston_paths(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_seed: int | None = None,
) -> HestonSimulationResult:
    """
        Given parameters, simulate n paths under the Heston model simulataneously with vectorization
        
        Model:
            dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW

        with corr(dW_1, dW_2) = rho
    """
    # Validate parameters
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if v0 < 0:
        raise ValueError("v0 must be non-negative.")
    if theta < 0:
        raise ValueError("theta must be non-negative.")
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")
    if xi < 0:
        raise ValueError("xi must be non-negative.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be between -1 and 1.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    rng = np.random.default_rng(random_seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    times = np.linspace(0.0, T, n_steps + 1)

   
    prices = np.zeros((n_paths, n_steps + 1))
    variances = np.zeros((n_paths, n_steps + 1))

    prices[:, 0] = S0
    variances[:, 0] = v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2_independent = rng.standard_normal(n_paths)

        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2_independent

        v_prev = np.maximum(variances[:, t], 0.0)
        sqrt_v_prev = np.sqrt(v_prev)

        # Variance update
        v_next = (
            v_prev
            + kappa * (theta - v_prev) * dt
            + xi * sqrt_v_prev * sqrt_dt * z2
        )
        v_next = np.maximum(v_next, 0.0)

        # Price update
        s_prev = prices[:, t]
        s_next = s_prev * np.exp(
            (mu - 0.5 * v_prev) * dt
            + sqrt_v_prev * sqrt_dt * z1
        )

        variances[:, t + 1] = v_next
        prices[:, t + 1] = s_next

    return HestonSimulationResult(
        times=times,
        prices=prices,
        variances=variances,
    )

def simulate_baseline_heston_paths(
    tickers: list[str],
    weights: dict[str, float],
    start: str,
    end: str,
    T: float,
    n_steps: int,
    window: int=21,
    annualization_factor: int=252,
    n_paths: int=1,
    portfolio_name: str = "portfolio",
    random_seed: int | None = None
) -> HestonSimulationResult:
    """
    Calibrate deterministic baseline Heston parameters from historical returns,
    then simulate Heston portfolio value and variance paths from those parameters.
    
    """
    
    params = calibrate_heston_params(tickers=tickers, weights=weights, start=start, end=end, window=window, annualization_factor=annualization_factor, portfolio_name=portfolio_name)
    
    result = generate_heston_paths(S0=params.S0, v0=params.v0, mu=params.mu, kappa=params.kappa, theta=params.theta, xi=params.xi, rho=params.rho, T=T, n_steps=n_steps, n_paths=n_paths, random_seed=random_seed)

    return HestonSimulationResult(
        times=result.times,
        prices=result.prices,
        variances=result.variances,
        params=params,
    )

def calibrate_heston_params(
    tickers: list[str],
    weights: dict[str,float],
    start: str,
    end: str,
    window: int = 21,
    annualization_factor: int=252,
    portfolio_name: str = "portfolio",
)->HestonCalibratedParams:
    """
    Calculate deterministic, constant parameters for the baseline Heston model from historical data
    Will be compared to the ML model which learns parameters from features
    """
    #Basic validation checks
    if not tickers:
        raise ValueError("tickers must be non-empty")
    tickers = [t.upper().strip() for t in tickers]
    if any(not t for t in tickers):
        raise ValueError("All tickers must be non-empty strings.")
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    
    #Fetch historical data
    price_history = fetch_price_history(tickers=tickers, start=start, end=end)
    prices = price_history.prices.copy()
    if prices.empty:
        raise ValueError("No price data returned for the requested portfolio.")
    port_p = portfolio_value_series(prices, weights)
    port_r = portfolio_log_returns_from_value(port_p)



    if len(port_r) < window + 5:
        raise ValueError(
            "Not enough return observations to compute realized volatility for portfolio."
        )
    
    #Rolling realized volatility and variance
    realized_vol = port_r.rolling(window=window).std() * np.sqrt(annualization_factor)
    realized_vol = realized_vol.dropna()

    if realized_vol.empty:
        raise ValueError(
            f"Realized volatility series is empty for portfolio. Try a longer date range."
        )
    
    realized_var = (realized_vol ** 2).dropna()

    if len(realized_var) < 3:
        raise ValueError(
            f"Not enough realized variance observations to calibrate Heston parameters for portfolio."
        )
    
    
    S0 = float(port_p.iloc[-1]) # Most recent portfolio value, initial level for simulation
    
    theta = float(realized_var.mean()) # Mean variance 
    v0 = float(realized_var.iloc[-1]) # Most recent variance

    dt = 1 / annualization_factor # Time step

    v_t = realized_var.iloc[:-1].to_numpy()
    v_next = realized_var.iloc[1:].to_numpy()
    delta_v = v_next - v_t
    x = (theta - v_t) * dt
    

    coeffs, _, _, _ = np.linalg.lstsq(x.reshape(-1, 1), delta_v, rcond=None)
    slope =  float(coeffs[0])
    kappa = max(slope,0.0) # Mean reversion strength

    denom = np.sqrt(np.maximum(v_t, 1e-12)) * np.sqrt(dt)

    residual = delta_v - kappa * x
    y_t = residual / denom
    xi = max(float(np.std(y_t, ddof=0)),0.0) # Vol-of-vol
 
    mu = float(port_r.mean() * annualization_factor) # Deterministic mean
    aligned_index = realized_var.index[:-1]
    log_returns = port_r.loc[aligned_index].to_numpy()
    z1 = (log_returns - (mu - 0.5 * v_t) * dt) / denom

    if xi > 1e-12:  
        z2 = residual / (xi * denom)
        if np.std(z1) < 1e-8 or np.std(z2) < 1e-8:
            rho = 0.0
        else:
            rho = np.corrcoef(z1, z2)[0, 1]
    else:
        rho = 0.0


    rho = float(np.clip(rho, -1.0, 1.0)) # Correlation between variance randomness and price randomness

   

    return HestonCalibratedParams(
        portfolio_name=portfolio_name,
        start = start,
        end = end,
        window = window,

        S0=S0,
        mu=mu,
        v0=v0,
        theta=theta,
        kappa=kappa,
        xi=xi,
        rho=rho,

        realized_vol=realized_vol,
        realized_var=realized_var,
        portfolio_returns=port_r,
        portfolio_prices= port_p,
    )


def main():
    import matplotlib.pyplot as plt

    fetch_start = "2017-01-01"
    fetch_end = "2026-03-27"
    tickers=["AAPL","NVDA","MSFT","GOOGL"]
    weights = {
        "AAPL": 0.25,
        "NVDA": 0.25,
        "MSFT": 0.25,
        "GOOGL": 0.25,
    }
    T=1.0
    n_steps=252
    annualization_factor = 252
    window=21
    n_paths=100
    
    result = simulate_baseline_heston_paths(
        tickers=tickers,
        weights=weights,
        start=fetch_start,
        end=fetch_end,
        T=T,
        n_steps=n_steps,
        window=window,
        annualization_factor=annualization_factor,
        n_paths=n_paths,
        random_seed=42,
        portfolio_name="Big_tech",
    )

    # ---------- Terminal summaries ----------
    params = result.params
    terminal_prices = result.prices[:, -1]
    terminal_variances = result.variances[:, -1]

    print("\n=== Baseline Heston Calibrated Parameters ===")
    print(f"Portfolio: {params.portfolio_name}")
    print(f"S0:     {params.S0:.4f}")
    print(f"mu:     {params.mu:.6f}")
    print(f"v0:     {params.v0:.6f}")
    print(f"theta:  {params.theta:.6f}")
    print(f"kappa:  {params.kappa:.6f}")
    print(f"xi:     {params.xi:.6f}")
    print(f"rho:    {params.rho:.6f}")

    print("\n=== Terminal Price Summary ===")
    print(f"Paths:                 {n_paths}")
    print(f"Mean terminal price:   {terminal_prices.mean():.4f}")
    print(f"Median terminal price: {np.median(terminal_prices):.4f}")
    print(f"Std terminal price:    {terminal_prices.std():.4f}")
    print(f"Min terminal price:    {terminal_prices.min():.4f}")
    print(f"Max terminal price:    {terminal_prices.max():.4f}")
    print(f"5th percentile:        {np.percentile(terminal_prices, 5):.4f}")
    print(f"95th percentile:       {np.percentile(terminal_prices, 95):.4f}")

    print("\n=== Terminal Variance Summary ===")
    print(f"Mean terminal var:     {terminal_variances.mean():.6f}")
    print(f"Median terminal var:   {np.median(terminal_variances):.6f}")
    print(f"Std terminal var:      {terminal_variances.std():.6f}")
    print(f"Min terminal var:      {terminal_variances.min():.6f}")
    print(f"Max terminal var:      {terminal_variances.max():.6f}")
    print(f"5th percentile:        {np.percentile(terminal_variances, 5):.6f}")
    print(f"95th percentile:       {np.percentile(terminal_variances, 95):.6f}")

    # ---------- Mean paths ----------
    mean_price_path = result.prices.mean(axis=0)
    mean_variance_path = result.variances.mean(axis=0)

    # ---------- Plot simulated price paths ----------
    plt.figure(figsize=(12, 6))
    for i in range(min(20, n_paths)):
        plt.plot(result.times, result.prices[i], alpha=0.35)
    plt.plot(result.times, mean_price_path, linewidth=2, label="Mean simulated price path")
    plt.axhline(params.S0, linestyle="--", linewidth=1, label="Initial price S0")
    plt.title(f"{params.portfolio_name} Baseline Heston Simulated Price Paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Plot simulated variance paths ----------
    plt.figure(figsize=(12, 6))
    for i in range(min(20, n_paths)):
        plt.plot(result.times, result.variances[i], alpha=0.35)
    plt.plot(result.times, mean_variance_path, linewidth=2, label="Mean simulated variance path")
    plt.axhline(params.theta, linestyle="--", linewidth=1, label="Long-run variance theta")
    plt.axhline(params.v0, linestyle=":", linewidth=1, label="Initial variance v0")
    plt.title(f"{params.portfolio_name} Baseline Heston Simulated Variance Paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Plot historical realized variance ----------
    plt.figure(figsize=(12, 6))
    plt.plot(params.realized_var.index, params.realized_var.values, label="Historical realized variance")
    plt.axhline(params.theta, linestyle="--", linewidth=1, label="theta")
    plt.axhline(params.v0, linestyle=":", linewidth=1, label="v0")
    plt.title(f"{params.portfolio_name} Historical Realized Variance vs Calibrated Levels")
    plt.xlabel("Date")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()