"""
Gain-to-Pain Ratio Calculation Utility
--------------------------------------
The Gain-to-Pain Ratio measures the cumulative return divided by the absolute
loss sum of a return series. It's used to identify strategies with smooth paths.
Created by Jack Schwager for his Market Wizards interviews.
"""

import numpy as np


def gain_to_pain_ratio(returns):
    """
    Computes the Gain-to-Pain Ratio.

    Args:
        returns (list or np.array): Series of returns.

    Returns:
        float: Gain-to-Pain Ratio.
    """
    returns = np.array(returns)
    total_returns = np.sum(returns)
    abs_loss_sum = np.abs(np.sum(returns[returns < 0]))

    if abs_loss_sum == 0:
        return 0

    return total_returns / abs_loss_sum


if __name__ == "__main__":
    # Simulate some normally distributed returns
    daily = np.random.normal(0.001, 0.01, 252)
    print(f"Gain-to-Pain Ratio: {gain_to_pain_ratio(daily):.4f}")
