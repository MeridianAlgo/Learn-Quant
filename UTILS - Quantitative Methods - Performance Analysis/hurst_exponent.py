"""
Hurst Exponent Calculation Utility
----------------------------------
The Hurst exponent (H) characterizes the long-term memory of a time series.

Interpretation:
- H < 0.5: Mean-reverting (anti-persistent) series
- H = 0.5: Random walk (Geometric Brownian Motion)
- H > 0.5: Trending (persistent) series
"""

import numpy as np


def compute_hurst(ts):
    """
    Computes the Hurst exponent of a time series using rescaled range (R/S) analysis.

    Args:
        ts (list or np.array): Time series of price data.

    Returns:
        float: Hurst exponent.
    """
    ts = np.array(ts)
    lags = range(2, 20)

    # Calculate the variance of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # The Hurst exponent is the slope of the fit
    return poly[0] * 2.0


if __name__ == "__main__":
    # Create a trending series
    trending = np.cumsum(np.random.randn(1000) + 0.1)
    # Create a mean-reverting series
    reverting = np.random.randn(1000)

    print(f"Trending Hurst: {compute_hurst(trending):.4f}")
    print(f"Mean-Reverting Hurst: {compute_hurst(reverting):.4f}")
