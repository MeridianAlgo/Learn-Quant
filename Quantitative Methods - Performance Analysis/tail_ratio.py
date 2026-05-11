"""
Tail Ratio Calculation Utility
------------------------------
The Tail Ratio measures the relationship between positive and negative outliers.
It's calculated as the absolute value of the 95th percentile (right tail)
divided by the absolute value of the 5th percentile (left tail).
"""

import numpy as np


def compute_tail_ratio(returns):
    """
    Computes the Tail Ratio.

    Args:
        returns (list or np.array): Series of returns.

    Returns:
        float: Tail Ratio.
    """
    returns = np.array(returns)
    right_tail = np.percentile(returns, 95)
    left_tail = np.abs(np.percentile(returns, 5))

    if left_tail == 0:
        return 0

    return right_tail / left_tail


if __name__ == "__main__":
    # Simulate some fat-tailed returns or normal ones
    daily = np.random.standard_t(df=5, size=1000) * 0.01
    print(f"Tail Ratio (T-Dist with df=5): {compute_tail_ratio(daily):.4f}")

    normal = np.random.normal(0.0005, 0.01, 1000)
    print(f"Tail Ratio (Normal): {compute_tail_ratio(normal):.4f}")
