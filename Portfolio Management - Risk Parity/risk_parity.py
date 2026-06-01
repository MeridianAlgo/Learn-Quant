"""
Risk Parity Portfolio Construction
-----------------------------------
Risk parity (also called Equal Risk Contribution, ERC) builds a portfolio in
which *every asset contributes the same amount of risk* to the total, rather
than the same amount of capital. This is the philosophy behind funds such as
Bridgewater's All Weather: instead of letting a few volatile assets dominate
the risk budget, each position is sized so its marginal impact on portfolio
volatility is equal.

Why it matters:
- A naive 60/40 stock/bond split is ~90% equity *risk* even though it is only
  60% equity *capital* — equities are far more volatile.
- Risk parity rebalances the risk, not the dollars, producing portfolios that
  are far more diversified in practice.

This module implements three increasingly sophisticated approaches:
- Inverse-volatility weighting   (ignores correlations, closed form)
- Equal Risk Contribution (ERC)  (accounts for correlations, solved numerically)
- Risk budgeting                 (ERC generalised to arbitrary target budgets)
"""

from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Annualised-units portfolio volatility (standard deviation).

    sigma_p = sqrt(w^T * Sigma * w)

    Args:
        weights: N-vector of portfolio weights.
        cov: NxN covariance matrix of asset returns.

    Returns:
        float: Portfolio volatility (same units as the covariance matrix).
    """
    weights = np.asarray(weights, dtype=float)
    # Quadratic form w' Sigma w is the portfolio variance; sqrt gives volatility.
    return float(np.sqrt(weights @ cov @ weights))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Component risk contribution of each asset to total portfolio volatility.

    The marginal risk contribution (MRC) is the partial derivative of portfolio
    volatility with respect to each weight:  MRC = (Sigma * w) / sigma_p.
    The *component* contribution scales that by the weight itself, so the
    contributions sum exactly to the portfolio volatility (Euler's theorem for
    homogeneous functions).

    Args:
        weights: N-vector of portfolio weights.
        cov: NxN covariance matrix.

    Returns:
        np.ndarray: N-vector of component risk contributions (sums to sigma_p).
    """
    weights = np.asarray(weights, dtype=float)
    sigma_p = portfolio_volatility(weights, cov)
    if sigma_p == 0:
        return np.zeros_like(weights)
    marginal = cov @ weights / sigma_p  # marginal risk contribution
    return weights * marginal  # component risk contribution


def inverse_volatility_weights(cov: np.ndarray) -> np.ndarray:
    """
    Naive risk parity: weight each asset by the inverse of its volatility.

    This is the closed-form solution to ERC *only when assets are uncorrelated*,
    but it is a fast, robust first approximation widely used in practice.

    Args:
        cov: NxN covariance matrix.

    Returns:
        np.ndarray: Long-only weights summing to 1.0.
    """
    vols = np.sqrt(np.diag(cov))
    inv = 1.0 / vols
    return inv / inv.sum()


def risk_parity_weights(
    cov: np.ndarray,
    budget: Optional[Union[list, np.ndarray]] = None,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Solve for Equal Risk Contribution (or arbitrary risk-budget) weights.

    We minimise the squared error between each asset's *fractional* risk
    contribution and its target budget. With the default budget (equal across
    assets) this yields the classic ERC portfolio.

    Args:
        cov: NxN covariance matrix.
        budget: Optional N-vector of target risk fractions (must sum to 1).
            Defaults to an equal budget (1/N each).
        max_iter: Maximum optimiser iterations.

    Returns:
        np.ndarray: Long-only weights summing to 1.0.
    """
    n = cov.shape[0]
    if budget is None:
        budget = np.ones(n) / n
    budget = np.asarray(budget, dtype=float)

    def objective(weights: np.ndarray) -> float:
        # Fractional risk contributions should match the target budget.
        rc = risk_contributions(weights, cov)
        sigma_p = rc.sum()
        if sigma_p == 0:
            return 1e9
        frac = rc / sigma_p
        return float(np.sum((frac - budget) ** 2))

    # Fully invested, long-only constraints.
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((1e-6, 1.0) for _ in range(n))
    # Start from inverse-vol weights — close to the solution, so converges fast.
    x0 = inverse_volatility_weights(cov)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )
    return result.x


if __name__ == "__main__":
    print("Risk Parity Portfolio Construction")
    print("=" * 40)

    # Three assets: a volatile equity, a mid-vol bond, and a low-vol cash proxy.
    vols = np.array([0.20, 0.10, 0.04])
    corr = np.array(
        [
            [1.00, 0.30, 0.05],
            [0.30, 1.00, 0.15],
            [0.05, 0.15, 1.00],
        ]
    )
    cov = np.outer(vols, vols) * corr  # build covariance from vols + correlation
    labels = ["Equity", "Bond", "Cash"]

    print("\nInverse-Volatility (naive) weights:")
    inv_w = inverse_volatility_weights(cov)
    for label, w in zip(labels, inv_w):
        print(f"  {label:7s}: {w:6.2%}")

    print("\nEqual Risk Contribution (ERC) weights:")
    erc_w = risk_parity_weights(cov)
    rc = risk_contributions(erc_w, cov)
    frac = rc / rc.sum()
    for label, w, f in zip(labels, erc_w, frac):
        print(f"  {label:7s}: weight={w:6.2%}  risk share={f:6.2%}")
    print(f"  Portfolio volatility: {portfolio_volatility(erc_w, cov):.4f}")

    # Custom risk budget: deliberately tilt 50% of risk into equities.
    print("\nRisk budgeting (50/30/20 risk split):")
    rb_w = risk_parity_weights(cov, budget=[0.5, 0.3, 0.2])
    rb_frac = risk_contributions(rb_w, cov)
    rb_frac /= rb_frac.sum()
    for label, w, f in zip(labels, rb_w, rb_frac):
        print(f"  {label:7s}: weight={w:6.2%}  risk share={f:6.2%}")
