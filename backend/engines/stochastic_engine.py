# For stochastic forecasting
import numpy as np

from dataclasses import asdict, is_dataclass, dataclass

from engines.heston_engine import generate_heston_paths
from engines.regime_engine import RegimeCalibratedParams, simulate_regime_switching_paths



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


def run_stochastic_forecast(*,model, s0, mu=None,  T, N, n, sigma=None,heston_params=None,garch_params=None,regime_params=None,random_seed=None):
    '''
        Single production entrypoint for stochastic engine
        Supports deterministic constant vol GBM, stochastic vol Heston model,
        GARCH(1,1), and GMM regime-switching Monte Carlo simulation.

        Expects that the parameters for each model have already been computed 
        and are provided here
    '''
    model = model.lower().strip()

    match model:
        case "gbm":
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
    
        case "heston":
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
    
        case "garch":
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

        case "regime":
            if regime_params is None:
                raise ValueError("regime_params is required when model='regime'.")
            if not isinstance(regime_params, RegimeCalibratedParams):
                raise TypeError("regime_params must be a RegimeCalibratedParams instance.")

            result = simulate_regime_switching_paths(
                start_regime_probs=regime_params.current_regime_probs,
                regime_stats=regime_params.regime_stats,
                transition_probs=regime_params.transition_probs,
                s0=s0,
                horizon=N,
                n_paths=n,
                random_seed=random_seed,
            )

            price_paths = result["prices"]
            variance_paths = result["variances"]
            model_params = {
                "model": "regime",
                "portfolio_name": regime_params.portfolio_name,
                "s0": float(s0),
                "n_regimes": int(regime_params.n_regimes),
                "current_regime_probs": [float(p) for p in regime_params.current_regime_probs],
                "T": float(T),
                "N": int(N),
                "n": int(n),
            }

        case _:
            raise ValueError("model must be one of 'gbm', 'heston', 'garch', or 'regime'.")

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