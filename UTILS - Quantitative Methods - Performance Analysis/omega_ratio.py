"""
Omega Ratio Calculation Utility
-------------------------------
The Omega Ratio measures the risk-adjusted return relative to a target return.
It's calculated as the probability-weighted gains divided by probability-weighted losses.
Used to identify the performance relative to a benchmark (minimum acceptable return).
"""

import numpy as np


def omega_ratio(returns, target=0):
    """
    Computes the Omega Ratio.

    Args:
        returns (list or np.array): Series of returns.
        target (float): Minimum acceptable return level.

    Returns:
        float: Omega Ratio.
    """
    returns = np.array(returns)
    gains = returns[returns > target] - target
    losses = target - returns[returns < target]

    if len(losses) == 0 or np.sum(losses) == 0:
        return np.inf

    return np.sum(gains) / np.sum(losses)


if __name__ == "__main__":
    # Simulate some normally distributed returns
    daily = np.random.normal(0.001, 0.01, 252)
    print(f"Omega Ratio at target 0%: {omega_ratio(daily):.4f}")
    print(f"Omega Ratio at target 0.1%: {omega_ratio(daily, target=0.001):.4f}")
