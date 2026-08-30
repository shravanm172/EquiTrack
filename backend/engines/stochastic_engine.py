# For stochastic forecasting
import numpy as np

from dataclasses import asdict, is_dataclass, dataclass

from engines.heston_engine import generate_heston_paths
from engines.regime_engine import RegimeCalibratedParams, simulate_regime_switching_paths
from engines.simulation_metrics import (
    summarize_terminal_metrics,
    summarize_drawdown_metrics,
    summarize_path_metrics,
    summarize_variance_metrics,
)



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