"""
Expected Shortfall (CVaR) Calculator
-------------------------------------
Expected Shortfall (ES), also known as Conditional Value at Risk (CVaR),
measures the expected loss given that the loss exceeds the VaR threshold.
ES is a coherent risk measure, unlike VaR.

Methods:
- Historical Simulation: Non-parametric, uses actual return distribution
- Parametric (Normal): Assumes normal distribution
- Cornish-Fisher: Adjusts for skewness and excess kurtosis
"""

import numpy as np
import scipy.stats as stats
from typing import Union


def historical_es(returns: Union[list, np.ndarray], confidence_level: float = 0.95) -> float:
    """
    Historical simulation Expected Shortfall.
    ES = mean of returns below the VaR threshold.

    Args:
        returns: Series of portfolio returns.
        confidence_level: Confidence level (e.g., 0.95 for 95%).

    Returns:
        float: ES as a positive number (magnitude of expected loss).
    """
    returns = np.array(returns)
    threshold = np.percentile(returns, (1 - confidence_level) * 100)
    tail_losses = returns[returns <= threshold]
    return float(-np.mean(tail_losses))


def parametric_es(returns: Union[list, np.ndarray], confidence_level: float = 0.95) -> float:
    """
    Parametric Expected Shortfall assuming normal distribution.
    ES = -(mu - sigma * phi(z) / (1 - confidence_level))

    Args:
        returns: Series of portfolio returns.
        confidence_level: Confidence level.

    Returns:
        float: Parametric ES as a positive number.
    """
    returns = np.array(returns)
    mu = np.mean(returns)
    sigma = np.std(returns)
    alpha = 1 - confidence_level
    z = stats.norm.ppf(alpha)
    # E[X | X < VaR] = mu - sigma * phi(z) / alpha  (z < 0, phi(z) > 0)
    es = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return float(es)


def cornish_fisher_es(returns: Union[list, np.ndarray], confidence_level: float = 0.95) -> float:
    """
    Cornish-Fisher ES adjusted for skewness and excess kurtosis.
    Provides better estimates for non-normal (fat-tailed) return distributions.

    Args:
        returns: Series of portfolio returns.
        confidence_level: Confidence level.

    Returns:
        float: Cornish-Fisher adjusted ES.
    """
    returns = np.array(returns)
    mu = np.mean(returns)
    sigma = np.std(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # excess kurtosis
    alpha = 1 - confidence_level
    z = stats.norm.ppf(alpha)

    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )
    var_cf = -(mu + sigma * z_cf)
    normal_var = -(mu + sigma * z)

    hist = historical_es(returns, confidence_level)
    if normal_var != 0:
        return float(hist * var_cf / normal_var)
    return hist


def es_summary(returns: Union[list, np.ndarray], confidence_level: float = 0.95) -> dict:
    """
    Returns all three ES estimates in a summary dictionary.

    Args:
        returns: Series of portfolio returns.
        confidence_level: Confidence level.

    Returns:
        dict: Historical, parametric, and Cornish-Fisher ES.
    """
    return {
        "historical_es": historical_es(returns, confidence_level),
        "parametric_es": parametric_es(returns, confidence_level),
        "cornish_fisher_es": cornish_fisher_es(returns, confidence_level),
        "confidence_level": confidence_level,
    }


if __name__ == "__main__":
    np.random.seed(42)
    daily_returns = stats.t.rvs(df=5, loc=0.001, scale=0.015, size=252)
    print("Expected Shortfall (CVaR) Analysis")
    print("=" * 40)
    summary = es_summary(daily_returns, confidence_level=0.95)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k.replace('_', ' ').title()}: {v:.4f}")
        else:
            print(f"{k.replace('_', ' ').title()}: {v}")
