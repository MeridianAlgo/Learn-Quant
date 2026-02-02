"""
Portfolio Optimizer: Mean-Variance Method
------------------------------------------
Find the best mix of assets for risk/return, using classic Markowitz Modern Portfolio Theory!
"""

import numpy as np


def mean_variance_optimizer(expected_returns, cov_matrix, risk_free_rate=0.0):
    """
    Calculates the tangency (maximum Sharpe ratio) portfolio weights.
    Args:
        expected_returns (np.array): Expected annual returns, shape (n_assets,)
        cov_matrix (np.array): Covariance matrix of returns, shape (n_assets, n_assets)
        risk_free_rate (float): Risk-free rate for Sharpe ratio (default 0)
    Returns:
        np.array: Optimal weights (add up to 1)
    """
    excess_returns = expected_returns - risk_free_rate
    inv_cov = np.linalg.inv(cov_matrix)
    weights = inv_cov @ excess_returns
    weights /= np.sum(weights)
    return weights


if __name__ == "__main__":
    # Example: Optimize a portfolio of 3 assets
    means = np.array([0.08, 0.10, 0.12])  # 8%, 10%, 12%
    cov = np.array(
        [[0.04, 0.01, 0.01], [0.01, 0.09, 0.02], [0.01, 0.02, 0.16]]
    )  # Covariance matrix
    w = mean_variance_optimizer(means, cov, risk_free_rate=0.03)
    print("Optimal Portfolio Weights:", w)
