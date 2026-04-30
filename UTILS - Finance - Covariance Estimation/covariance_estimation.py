"""
Robust Covariance Estimation
------------------------------
Standard sample covariance is noisy and ill-conditioned with many assets.
Shrinkage estimators blend sample covariance with a structured target.

Methods:
- Sample Covariance: MLE estimate, noisy in high dimensions
- Ledoit-Wolf Analytical: Oracle approximating shrinkage toward scaled identity
- Constant Correlation: Shrink toward equal-correlation matrix
- EWMA (RiskMetrics): Downweights old observations
"""

import numpy as np


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """
    Standard sample covariance matrix.

    Args:
        returns: T×N matrix of returns (T observations, N assets).

    Returns:
        np.ndarray: N×N sample covariance matrix.
    """
    return np.cov(returns.T)


def ledoit_wolf_shrinkage(returns: np.ndarray) -> dict:
    """
    Analytical Ledoit-Wolf shrinkage toward scaled identity.
    Sigma* = (1 - alpha) * S + alpha * mu_S * I

    Args:
        returns: T×N returns matrix.

    Returns:
        dict: shrunk_cov, alpha (shrinkage intensity), target.
    """
    T, N = returns.shape
    S = sample_covariance(returns)
    mu = np.trace(S) / N
    S_target = mu * np.eye(N)

    returns_centered = returns - returns.mean(axis=0)
    phi_hat = 0.0
    for t in range(T):
        x = returns_centered[t, :].reshape(-1, 1)
        phi_hat += np.linalg.norm(x @ x.T - S, "fro") ** 2
    phi_hat /= T**2

    target_dist_sq = np.linalg.norm(S - S_target, "fro") ** 2
    alpha = float(np.clip(phi_hat / target_dist_sq if target_dist_sq > 0 else 0.0, 0.0, 1.0))
    shrunk = (1 - alpha) * S + alpha * S_target

    return {"shrunk_cov": shrunk, "alpha": alpha, "target": S_target}


def constant_correlation_shrinkage(returns: np.ndarray) -> dict:
    """
    Shrink sample covariance toward constant correlation matrix.
    Preserves sample variances; equalizes all pairwise correlations to mean.

    Args:
        returns: T×N returns matrix.

    Returns:
        dict: shrunk_cov, alpha, mean_correlation.
    """
    T, N = returns.shape
    S = sample_covariance(returns)
    std_devs = np.sqrt(np.diag(S))
    corr = S / np.outer(std_devs, std_devs)

    upper_tri = corr[np.triu_indices(N, k=1)]
    r_bar = float(np.mean(upper_tri))

    target_corr = r_bar * np.ones((N, N)) + (1 - r_bar) * np.eye(N)
    target = np.outer(std_devs, std_devs) * target_corr

    num = float(np.sum((S - target) ** 2))
    den = float(np.sum(S**2) - np.sum(np.diag(S) ** 2))
    alpha = float(np.clip((T - 2) / T * num / den if den > 0 else 0.0, 0.0, 1.0))
    shrunk = (1 - alpha) * S + alpha * target

    return {"shrunk_cov": shrunk, "alpha": alpha, "mean_correlation": r_bar}


def ewma_covariance(returns: np.ndarray, lambda_: float = 0.94) -> np.ndarray:
    """
    Exponentially Weighted Moving Average covariance (RiskMetrics).
    Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_{t-1} * r_{t-1}'

    Args:
        returns: T×N returns matrix.
        lambda_: Decay factor (0.94 for daily, 0.97 for monthly).

    Returns:
        np.ndarray: EWMA covariance matrix (N×N).
    """
    T, N = returns.shape
    warmup = min(21, T // 4)
    cov = np.cov(returns[:warmup].T) if warmup > 1 else np.eye(N) * 0.01

    for t in range(warmup, T):
        r = returns[t].reshape(-1, 1)
        cov = lambda_ * cov + (1 - lambda_) * (r @ r.T)

    return cov


def condition_number(cov: np.ndarray) -> float:
    """
    Condition number of covariance matrix. Lower = better conditioned.

    Returns:
        float: Ratio of largest to smallest eigenvalue.
    """
    eigenvalues = np.linalg.eigvalsh(cov)
    min_eig = eigenvalues[0]
    if min_eig <= 0:
        return np.inf
    return float(eigenvalues[-1] / min_eig)


if __name__ == "__main__":
    np.random.seed(42)
    T, N = 252, 10
    returns = np.random.multivariate_normal(
        mean=np.zeros(N),
        cov=np.eye(N) * 0.01,
        size=T,
    )

    print("Covariance Estimation Comparison")
    print("=" * 45)

    S = sample_covariance(returns)
    lw = ledoit_wolf_shrinkage(returns)
    cc = constant_correlation_shrinkage(returns)
    ew = ewma_covariance(returns)

    print(f"Sample covariance condition number:     {condition_number(S):.2f}")
    print(f"Ledoit-Wolf condition number:           {condition_number(lw['shrunk_cov']):.2f}  (alpha={lw['alpha']:.3f})")
    print(f"Constant correlation condition number:  {condition_number(cc['shrunk_cov']):.2f}  (alpha={cc['alpha']:.3f})")
    print(f"EWMA condition number:                  {condition_number(ew):.2f}")
