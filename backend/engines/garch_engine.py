from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

from scipy.optimize import minimize

@dataclass 
class GarchCalibratedParams:
    mu:float
    omega:float
    alpha:float
    beta:float
    h0:float

def _compute_garch_variance_path(returns:np.ndarray, mu:float, omega:float, alpha:float, beta:float, h0:float,)->np.ndarray:
    """
        Helper function to generate the variance path for a given candidate parameter set
        Will be used alongside observed shocks to compute likelihoods for MLE optimization
    """
    n = len(returns)
    h = np.empty(n, dtype=float)
    h[0] = max(h0, 1e-12)

    for t in range(1,n):
        eps_prev = returns[t-1] - mu
        h[t] = omega + alpha * (eps_prev ** 2) + beta * h[t - 1]
        h[t] = max(h[t], 1e-12)
    
    return h


def _garch_negative_log_likelihood(
        params:np.ndarray,
        returns:np.ndarray,
        estimate_mu:bool,
)->float:
    """
        Calculate negative log-likelihood for GARCH(1,1) under conditional normality

        Note: h0 is not being optimized here

        If estimate_mu=True:
            params = [mu, omega, alpha, beta]
         else:
            params = [omega, alpha, beta] and mu = 0
    """
    if estimate_mu:
        mu, omega, alpha, beta = params
    else:
        mu = 0.0
        omega, alpha, beta = params

    # enforce hard constraints
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999999:
        return 1e20
    
    # choose good starting point for h0 from historical returns
    sample_var = float(np.var(returns, ddof=1))
    h0 = max(sample_var, 1e-12)

    # generate modelled variance path
    h = _compute_garch_variance_path(
        returns=returns,
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        h0=h0,
    )

    # observed shocks
    eps = returns - mu

    # negative log-likelihood
    nll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h) + (eps ** 2) / h)

    if not np.isfinite(nll):
        return 1e20

    return float(nll)

def calibrate_garch_mle(
        returns:pd.Series,
        estimate_mu:bool=True,
)->GarchCalibratedParams:
    """
        Fit historically optimized GARCH(1,1) parameters by MLE
        returns: 1D array of historical returns
    """
    returns = np.asarray(returns.to_numpy(), dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if len(returns) < 30:
        raise ValueError("need at least 30 historical returns to calibrate GARCH reliably")
    
    sample_mean = float(np.mean(returns))
    sample_var = float(np.var(returns, ddof=1))
    sample_var = max(sample_var, 1e-12)

    # reasonable initial guess for candidate parameter set
    omega0 = sample_var * 0.01
    alpha0 = 0.05
    beta0 = 0.9

    if estimate_mu:
        x0 = np.array([sample_mean, omega0, alpha0, beta0], dtype=float)
        bounds = [
            (None, None),       # mu
            (1e-12, None),      # omega
            (0.0, 0.999),       # alpha
            (0.0, 0.999),       # beta
        ]
    else:
        x0 = np.array([omega0, alpha0, beta0], dtype=float)
        bounds = [
            (1e-12, None),
            (0.0, 0.999),
            (0.0, 0.999),
        ]

    result = minimize(
        fun=_garch_negative_log_likelihood,
        x0=x0,
        args=(returns, estimate_mu),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    
    if not result.success:
        raise RuntimeError(f"GARCH calibration failed: {result.message}")

    if estimate_mu:
        mu, omega, alpha, beta = result.x
    else:
        mu = 0.0
        omega, alpha, beta = result.x

    h0 = sample_var

    return GarchCalibratedParams(
        mu=float(mu),
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        h0=float(h0),
    )
