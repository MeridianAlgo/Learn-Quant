"""
Active Performance Metrics Utility
----------------------------------
Information Ratio and Tracking Error measure a portfolio manager's skill
at outperforming a benchmark.

- Tracking Error: Standard deviation of active returns.
- Information Ratio: Active return divided by Tracking Error.
"""

import numpy as np


def active_metrics(returns, benchmark_returns):
    """
    Computes Tracking Error and Information Ratio.

    Args:
        returns (list or np.array): Series of portfolio returns.
        benchmark_returns (list or np.array): Series of benchmark returns.

    Returns:
        dict: A dictionary containing 'tracking_error' and 'information_ratio'.
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)

    if len(returns) != len(benchmark_returns):
        raise ValueError("Return series and benchmark series must be the same length.")

    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns)

    if tracking_error == 0:
        return {"tracking_error": 0, "information_ratio": 0}

    information_ratio = np.mean(active_returns) / tracking_error

    return {"tracking_error": tracking_error, "information_ratio": information_ratio}


if __name__ == "__main__":
    # Simulate a benchmark index (e.g. S&P 500)
    benchmark = np.random.normal(0.0005, 0.01, 252)
    # Simulate an active manager with some "alpha"
    alpha = 0.0001
    portfolio = benchmark + alpha + np.random.normal(0, 0.002, 252)

    stats = active_metrics(portfolio, benchmark)
    for k, v in stats.items():
        print(f"{k.replace('_', ' ').title()}: {v:.4f}")
