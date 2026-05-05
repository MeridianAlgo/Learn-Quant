"""
GARCH Volatility Models
------------------------
Generalized Autoregressive Conditional Heteroskedasticity models capture
volatility clustering — periods of high vol followed by high vol, low by low.

Implements GARCH(1,1) MLE estimation, EWMA, and one-step + multi-step forecasts.
"""

from typing import Optional, Union

import numpy as np


def ewma_volatility(
    returns: Union[list, np.ndarray],
    lambda_: float = 0.94,
) -> np.ndarray:
    """
    Exponentially Weighted Moving Average volatility (RiskMetrics 1996).

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

    Args:
        returns: Return series.
        lambda_: Decay factor (0.94 daily, 0.97 monthly per RiskMetrics).

    Returns:
        np.ndarray: Conditional standard deviation series.
    """
    r = np.array(returns, dtype=float)
    n = len(r)
    var = np.full(n, np.nan)
    var[0] = float(np.var(r))
    for t in range(1, n):
        var[t] = lambda_ * var[t - 1] + (1.0 - lambda_) * r[t - 1] ** 2
    return np.sqrt(var)


def garch_log_likelihood(
    params: np.ndarray,
    returns: np.ndarray,
) -> float:
    """
    Negative log-likelihood for GARCH(1,1) under Gaussian innovations.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    """
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10

    n = len(returns)
    sigma2 = np.empty(n)
    sigma2[0] = float(np.var(returns))
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    if np.any(sigma2 <= 0):
        return 1e10

    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    return -ll


def fit_garch(
    returns: Union[list, np.ndarray],
    init: Optional[np.ndarray] = None,
) -> dict:
    """
    Fit GARCH(1,1) by Maximum Likelihood (Gaussian).

    Args:
        returns: Return series (mean-centered recommended).
        init: Initial [omega, alpha, beta]. Defaults to [var*0.05, 0.1, 0.85].

    Returns:
        dict: omega, alpha, beta, sigma (conditional std), log_lik, persistence.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as err:
        raise ImportError("scipy required: pip install scipy") from err

    r = np.array(returns, dtype=float)
    r = r - np.mean(r)
    var0 = float(np.var(r))

    if init is None:
        init = np.array([var0 * 0.05, 0.10, 0.85])

    bounds = [(1e-12, None), (1e-6, 1.0), (1e-6, 0.999)]
    res = minimize(
        garch_log_likelihood,
        init,
        args=(r,),
        method="L-BFGS-B",
        bounds=bounds,
    )

    omega, alpha, beta = res.x
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = var0
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma": np.sqrt(sigma2),
        "log_lik": float(-res.fun),
        "persistence": float(alpha + beta),
        "unconditional_vol": float(np.sqrt(omega / max(1.0 - alpha - beta, 1e-12))),
        "converged": bool(res.success),
    }


def garch_forecast(
    fit: dict,
    last_return: float,
    horizon: int = 10,
) -> np.ndarray:
    """
    Multi-step variance forecast from a fitted GARCH(1,1).

    sigma_{t+h}^2 = omega * (1 + (a+b) + ... + (a+b)^{h-1}) + (a+b)^h * sigma_t^2

    Args:
        fit: Output of fit_garch().
        last_return: Most recent return r_t.
        horizon: Forecast horizon in periods.

    Returns:
        np.ndarray: sigma forecast for h=1..horizon.
    """
    omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
    sigma2_t = fit["sigma"][-1] ** 2
    persistence = alpha + beta

    forecasts = np.empty(horizon)
    sigma2_h = omega + alpha * last_return**2 + beta * sigma2_t
    forecasts[0] = sigma2_h
    for h in range(1, horizon):
        sigma2_h = omega + persistence * sigma2_h
        forecasts[h] = sigma2_h
    return np.sqrt(forecasts)


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    omega_t, alpha_t, beta_t = 1e-5, 0.08, 0.90
    r = np.zeros(n)
    sigma2 = np.full(n, 1e-4)
    for t in range(1, n):
        sigma2[t] = omega_t + alpha_t * r[t - 1] ** 2 + beta_t * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * np.random.normal()

    print("GARCH(1,1) Fit")
    print("=" * 40)
    fit = fit_garch(r)
    print(f"omega       = {fit['omega']:.6e}")
    print(f"alpha       = {fit['alpha']:.4f}  (true: {alpha_t})")
    print(f"beta        = {fit['beta']:.4f}  (true: {beta_t})")
    print(f"persistence = {fit['persistence']:.4f}")
    print(f"uncond vol  = {fit['unconditional_vol']:.4f}")

    fc = garch_forecast(fit, r[-1], horizon=5)
    print(f"\n5-step vol forecast: {fc}")
