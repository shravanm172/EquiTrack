from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

from scipy.optimize import minimize
from scipy.special import gammaln

@dataclass 
class GarchCalibratedParams:
    mu:float
    omega:float
    alpha:float
    beta:float
    nu:float
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
        mu, log_omega, alpha, beta, nu = params
    else:
        mu = 0.0
        log_omega, alpha, beta, nu = params

    omega = np.exp(log_omega)

    # enforce hard constraints
    if not np.isfinite(omega) or omega <= 0.0 or alpha < 0.0 or beta < 0.0 or nu <= 2.0:
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
    h = np.maximum(h, 1e-12)
    z2 = (eps ** 2) / h

    c = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log((nu - 2) * np.pi)
    )

    loglik = c - 0.5 * np.log(h) - ((nu + 1) / 2) * np.log(1 + z2 / (nu - 2))

    nll = -np.sum(loglik)

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
    omega0 = max(sample_var * 0.01, 1e-12)
    log_omega0 = np.log(omega0)
    alpha0 = 0.05
    beta0 = 0.9
    nu0 = 8.0

    if estimate_mu:
        x0 = np.array([sample_mean, log_omega0, alpha0, beta0, nu0], dtype=float)
        bounds = [
            (None, None),       # mu
            (-50.0, 10.0),      # omega
            (0.0, 0.999),       # alpha
            (0.0, 0.999),       # beta
            (2.01, 30),         # nu
        ]
        constraints = [
            {
                "type": "ineq",
                "fun": lambda x: 0.999999 - (x[2] + x[3])
            }
        ]
    else:
        x0 = np.array([log_omega0, alpha0, beta0, nu0], dtype=float)
        bounds = [
            (-50.0, 10.0),
            (0.0, 0.999),
            (0.0, 0.999),
            (2.01, 30),     
        ]
        constraints = [
            {
                "type": "ineq",
                "fun": lambda x: 0.999999 - (x[1] + x[2])
            }
        ]

    # print("\n=== LOCAL OBJECTIVE CHECK ===")
    # print("x0:", x0)
    # base = _garch_negative_log_likelihood(x0, returns, estimate_mu)
    # print("nll(x0):", base)

    # for i in range(len(x0)):
    #     x_test_up = x0.copy()
    #     x_test_dn = x0.copy()

    #     x_test_up[i] += 1e-3
    #     x_test_dn[i] -= 1e-3

    #     val_up = _garch_negative_log_likelihood(x_test_up, returns, estimate_mu)
    #     val_dn = _garch_negative_log_likelihood(x_test_dn, returns, estimate_mu)

    #     print(f"param {i}: up -> {val_up}, down -> {val_dn}, "
    #         f"delta_up -> {val_up - base}, delta_down -> {val_dn - base}")
    # print("=============================\n")

    
    
    
    result = minimize(
        fun=_garch_negative_log_likelihood,
        x0=x0,
        args=(returns, estimate_mu),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    print("\n=== OPTIMIZER RESULT ===")
    print("success:", result.success)
    print("message:", result.message)
    print("nit:", result.nit)
    print("nfev:", result.nfev)
    print("njev:", result.njev)
    print("x0:", x0)
    print("x*:", result.x)
    print("fun:", result.fun)
    print("jac:", result.jac)
    print("========================\n")

    
    if not result.success:
        raise RuntimeError(f"GARCH calibration failed: {result.message}")

    if estimate_mu:
        mu, log_omega, alpha, beta, nu = result.x
    else:
        mu = 0.0
        log_omega, alpha, beta, nu = result.x

    omega = float(np.exp(log_omega))
    h0 = sample_var

    # print("\n=== CALIBRATION RETURN ===")
    # print("mu:", mu)
    # print("omega:", omega)
    # print("alpha:", alpha)
    # print("beta:", beta)
    # print("nu:", nu)
    # print("==========================\n")

    return GarchCalibratedParams(
        mu=float(mu),
        omega=omega,
        alpha=float(alpha),
        beta=float(beta),
        h0=float(h0),
        nu=float(nu),
    )
