"""
Black-Litterman Portfolio Optimization
----------------------------------------
The Black-Litterman model combines market equilibrium returns with
investor views to produce more stable and intuitive optimal portfolios.

Steps:
1. Start with market-implied (equilibrium) returns via reverse optimization
2. Express investor views as P*mu = Q with uncertainty Omega
3. Blend equilibrium and views using Bayesian update
4. Optimize on blended returns
"""

import numpy as np
from typing import Optional


def market_implied_returns(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """
    Reverse optimization: compute implied returns from market portfolio.
    Pi = lambda * Sigma * w_mkt

    Args:
        cov_matrix: NxN covariance matrix of returns.
        market_weights: N-vector of market cap weights.
        risk_aversion: Risk aversion coefficient (lambda).

    Returns:
        np.ndarray: N-vector of implied equilibrium excess returns.
    """
    return risk_aversion * cov_matrix @ market_weights


def black_litterman(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: Optional[np.ndarray] = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> dict:
    """
    Black-Litterman model: blend equilibrium returns with views.

    Args:
        cov_matrix: NxN covariance matrix.
        market_weights: N-vector market cap weights.
        P: KxN pick matrix (K views, N assets). Row i encodes view i.
        Q: K-vector of view returns.
        omega: KxK uncertainty matrix of views. If None, uses proportional to P*tau*Sigma*P'.
        tau: Scaling parameter (typically 0.01-0.05).
        risk_aversion: Risk aversion lambda.

    Returns:
        dict: posterior_returns, posterior_covariance, equilibrium_returns.
    """
    pi = market_implied_returns(cov_matrix, market_weights, risk_aversion)
    tau_sigma = tau * cov_matrix

    if omega is None:
        omega = np.diag(np.diag(P @ tau_sigma @ P.T))

    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(omega)

    posterior_cov_inv = tau_sigma_inv + P.T @ omega_inv @ P
    posterior_cov = np.linalg.inv(posterior_cov_inv)
    posterior_mean = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    return {
        "posterior_returns": posterior_mean,
        "posterior_covariance": posterior_cov + cov_matrix,
        "equilibrium_returns": pi,
    }


def bl_optimal_weights(
    bl_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """
    Mean-variance optimal weights given BL posterior returns.
    w* = (1/lambda) * Sigma^{-1} * mu_BL, normalized to sum-of-abs = 1.

    Args:
        bl_returns: N-vector of BL posterior expected returns.
        cov_matrix: NxN covariance matrix.
        risk_aversion: Risk aversion coefficient.

    Returns:
        np.ndarray: Optimal portfolio weights.
    """
    weights = np.linalg.inv(cov_matrix) @ bl_returns / risk_aversion
    return weights / np.sum(np.abs(weights))


if __name__ == "__main__":
    np.random.seed(42)
    asset_names = ["US Equity", "Int'l Equity", "Bonds", "Commodities"]

    corr = np.array([
        [1.00, 0.75, -0.20, 0.30],
        [0.75, 1.00, -0.15, 0.35],
        [-0.20, -0.15, 1.00, -0.05],
        [0.30, 0.35, -0.05, 1.00],
    ])
    vols = np.array([0.16, 0.18, 0.05, 0.20])
    cov = np.outer(vols, vols) * corr
    mkt_weights = np.array([0.40, 0.30, 0.20, 0.10])

    # View: US Equity outperforms Int'l Equity by 2%
    P = np.array([[1, -1, 0, 0]])
    Q = np.array([0.02])

    result = black_litterman(cov, mkt_weights, P, Q)
    opt_weights = bl_optimal_weights(result["posterior_returns"], result["posterior_covariance"])

    print("Black-Litterman Results")
    print("=" * 40)
    print("\nEquilibrium vs BL Returns:")
    for i, name in enumerate(asset_names):
        eq = result["equilibrium_returns"][i]
        bl = result["posterior_returns"][i]
        print(f"  {name:15s}: Eq={eq:.3f}  BL={bl:.3f}")
    print("\nOptimal Weights:")
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {opt_weights[i]:.3f}")
