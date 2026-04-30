"""
Kelly Criterion Position Sizing
---------------------------------
The Kelly Criterion determines the optimal fraction of capital to allocate
to maximize long-run geometric growth rate.

Methods:
- Discrete Kelly: For win/loss bets (f = p - q/b)
- Continuous Kelly: For continuous return distributions (f = mu/sigma^2)
- Fractional Kelly: Reduce full Kelly for variance control
- Multi-Asset Kelly: Portfolio-level optimal allocation
"""

import numpy as np
from typing import Union


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """
    Discrete Kelly Criterion for binary outcomes.
    f* = p - q/b  where p=win prob, q=loss prob, b=win/loss ratio.

    Args:
        win_prob: Probability of winning (0 < p < 1).
        win_loss_ratio: Ratio of win size to loss size (b/a).

    Returns:
        float: Kelly fraction (negative = bet opposite side; 0 = don't bet).

    Raises:
        ValueError: If inputs are invalid.
    """
    if not 0 < win_prob < 1:
        raise ValueError("win_prob must be in (0, 1)")
    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio must be positive")
    loss_prob = 1 - win_prob
    return float(win_prob - loss_prob / win_loss_ratio)


def kelly_continuous(mu: float, sigma: float) -> float:
    """
    Continuous Kelly for log-normal return distributions.
    f* = mu / sigma^2

    Args:
        mu: Expected (mean) return per period.
        sigma: Standard deviation of returns per period.

    Returns:
        float: Kelly fraction.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(mu / sigma**2)


def fractional_kelly(
    win_prob: float,
    win_loss_ratio: float,
    fraction: float = 0.5,
) -> float:
    """
    Fractional Kelly: scale full Kelly down to reduce variance.
    Common fractions: 0.25 (quarter-Kelly), 0.5 (half-Kelly).

    Args:
        win_prob: Probability of winning.
        win_loss_ratio: Win-to-loss ratio.
        fraction: Fraction of full Kelly (default 0.5).

    Returns:
        float: Fractional Kelly bet size.
    """
    return float(fraction * kelly_fraction(win_prob, win_loss_ratio))


def multi_asset_kelly(
    expected_returns: Union[list, np.ndarray],
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Multi-asset Kelly: f* = Sigma^{-1} * mu.
    Maximizes expected log return of a portfolio.

    Args:
        expected_returns: N-vector of expected returns.
        cov_matrix: NxN covariance matrix.

    Returns:
        np.ndarray: Kelly optimal weights (may include leverage > 1.0).
    """
    mu = np.array(expected_returns)
    return np.linalg.inv(cov_matrix) @ mu


def kelly_growth_rate(
    win_prob: float,
    win_loss_ratio: float,
    fraction: float = 1.0,
) -> float:
    """
    Expected logarithmic growth rate for a given Kelly fraction.
    g = p*ln(1 + f*b) + q*ln(1 - f)

    Args:
        win_prob: Probability of winning.
        win_loss_ratio: Win-to-loss ratio.
        fraction: Fraction of Kelly to use (1.0 = full Kelly).

    Returns:
        float: Expected log growth per period.
    """
    f = kelly_fraction(win_prob, win_loss_ratio) * fraction
    loss_prob = 1 - win_prob
    # Guard against log(0) for edge cases
    win_term = win_prob * np.log(max(1 + f * win_loss_ratio, 1e-12))
    loss_term = loss_prob * np.log(max(1 - f, 1e-12))
    return float(win_term + loss_term)


if __name__ == "__main__":
    print("Kelly Criterion Examples")
    print("=" * 40)

    # Discrete: 60% win rate, 2:1 payoff
    f_full = kelly_fraction(0.60, 2.0)
    f_half = fractional_kelly(0.60, 2.0, 0.5)
    print(f"\nDiscrete Kelly (60% win, 2:1 payoff):")
    print(f"  Full Kelly:  {f_full:.3f} ({f_full*100:.1f}% of capital)")
    print(f"  Half Kelly:  {f_half:.3f} ({f_half*100:.1f}% of capital)")

    print("\nGrowth rate vs Kelly fraction:")
    for frac in [0.25, 0.5, 0.75, 1.0, 1.25]:
        g = kelly_growth_rate(0.60, 2.0, frac)
        print(f"  {frac:.2f}x Kelly: {g:.5f} per period")

    mu, sigma = 0.0008, 0.015
    f_cont = kelly_continuous(mu, sigma)
    print(f"\nContinuous Kelly (mu={mu}, sigma={sigma}):")
    print(f"  Optimal fraction: {f_cont:.3f}")

    np.random.seed(42)
    mu_vec = np.array([0.10, 0.15, 0.08])
    cov = np.array([[0.04, 0.01, 0.005],
                    [0.01, 0.09, 0.008],
                    [0.005, 0.008, 0.02]])
    kelly_weights = multi_asset_kelly(mu_vec, cov)
    print("\nMulti-Asset Kelly Weights:")
    for i, w in enumerate(kelly_weights):
        print(f"  Asset {i+1}: {w:.3f}")
