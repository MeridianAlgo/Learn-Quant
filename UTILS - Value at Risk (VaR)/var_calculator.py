"""
Value at Risk (VaR) Calculator
-------------------------------
This module calculates Value at Risk using the basic (parametric) approach, helping users estimate potential portfolio losses.
"""

import numpy as np
import scipy.stats as stats


def value_at_risk(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) for a return series using the parametric (normal) method.
    Args:
        returns (list or np.array): Series of returns
        confidence_level (float): Confidence level (default 0.95)
    Returns:
        float: Daily VaR (as a positive number)
    """
    returns = np.array(returns)
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence_level
    var = -(mu + sigma * stats.norm.ppf(alpha))
    return var


if __name__ == "__main__":
    # Example usage
    daily_returns = np.random.normal(0.001, 0.02, 252)  # Simulate daily returns
    var95 = value_at_risk(daily_returns, 0.95)
    print(f"Daily VaR at 95% confidence: {var95:.4f}")
