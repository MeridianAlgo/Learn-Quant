"""
Optimal Execution, Almgren and Chriss
-------------------------------------
You need to sell a large block of stock before the close. Sell it all at once
and you crush the price with your own order. Sell it slowly and the price may
wander away from you while you wait. Every execution schedule sits somewhere
on that trade off between impact cost and timing risk.

Almgren and Chriss (2000) solved it in closed form for a simple but useful
model. Price impact is linear in trading speed, prices follow arithmetic
Brownian motion, and the trader minimises

    expected cost + lambda * variance of cost

The answer is a hyperbolic sine curve. With no risk aversion it is a straight
line, the TWAP schedule. As risk aversion rises the curve bends and more of
the order gets done early.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionProblem:
    """Inputs to the Almgren Chriss model.

    shares      total shares to liquidate
    horizon     time available, in days
    periods     number of trading intervals
    sigma       price volatility, dollars per share per sqrt(day)
    eta         temporary impact, dollars per share per (shares per day)
    gamma       permanent impact, dollars per share per share
    epsilon     fixed cost per share, for example half the spread
    """

    shares: float
    horizon: float
    periods: int
    sigma: float
    eta: float
    gamma: float = 0.0
    epsilon: float = 0.0

    def __post_init__(self):
        if self.shares <= 0:
            raise ValueError("shares must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.periods < 1:
            raise ValueError("periods must be at least 1")
        if self.sigma < 0 or self.eta <= 0 or self.gamma < 0 or self.epsilon < 0:
            raise ValueError("sigma, gamma and epsilon must be >= 0 and eta > 0")
        if self.eta_tilde <= 0:
            raise ValueError("eta - gamma * tau / 2 must be positive, use more periods or less permanent impact")

    @property
    def tau(self) -> float:
        return self.horizon / self.periods

    @property
    def eta_tilde(self) -> float:
        return self.eta - 0.5 * self.gamma * self.tau


def urgency(problem: ExecutionProblem, risk_aversion: float) -> float:
    """The decay rate kappa of the optimal schedule.

    Solves ``cosh(kappa * tau) = 1 + kappa_tilde^2 * tau^2 / 2`` with
    ``kappa_tilde^2 = lambda * sigma^2 / eta_tilde``. ``1 / kappa`` is the
    half life of the trade, roughly how long it takes to work off a large
    part of the position.
    """
    if risk_aversion < 0:
        raise ValueError("risk aversion must be >= 0")
    kt2 = risk_aversion * problem.sigma**2 / problem.eta_tilde
    tau = problem.tau
    return math.acosh(1.0 + 0.5 * kt2 * tau * tau) / tau


def optimal_trajectory(problem: ExecutionProblem, risk_aversion: float) -> np.ndarray:
    """Shares still held at each of the ``periods + 1`` decision times."""
    kappa = urgency(problem, risk_aversion)
    t = np.arange(problem.periods + 1) * problem.tau
    if kappa * problem.horizon < 1e-10:
        return problem.shares * (1.0 - t / problem.horizon)
    return problem.shares * np.sinh(kappa * (problem.horizon - t)) / np.sinh(kappa * problem.horizon)


def twap_trajectory(problem: ExecutionProblem) -> np.ndarray:
    """Straight line schedule, the zero risk aversion case."""
    return optimal_trajectory(problem, 0.0)


def trade_list(trajectory) -> np.ndarray:
    """Shares sold in each interval, ``n_k = x_k-1 - x_k``."""
    x = np.asarray(trajectory, dtype=float)
    if x.size < 2:
        raise ValueError("trajectory needs at least two points")
    return x[:-1] - x[1:]


def expected_cost(problem: ExecutionProblem, trajectory) -> float:
    """Expected implementation shortfall in dollars.

    ``0.5 * gamma * X^2 + epsilon * sum |n_k| + (eta_tilde / tau) * sum n_k^2``.
    The last term is why slow trading is cheaper. It is the sum of squares,
    and spreading the same total over more intervals shrinks it.
    """
    n = trade_list(trajectory)
    return float(
        0.5 * problem.gamma * problem.shares**2
        + problem.epsilon * np.abs(n).sum()
        + problem.eta_tilde / problem.tau * np.sum(n * n)
    )


def cost_variance(problem: ExecutionProblem, trajectory) -> float:
    """Variance of the shortfall, ``sigma^2 * tau * sum x_k^2``.

    Every share still held is exposed to the price for one more interval, so
    holding more for longer means more risk.
    """
    x = np.asarray(trajectory, dtype=float)
    return float(problem.sigma**2 * problem.tau * np.sum(x[1:] ** 2))


def efficient_frontier(problem: ExecutionProblem, risk_aversions) -> list[tuple[float, float, float]]:
    """``(lambda, expected cost, standard deviation)`` for each risk aversion."""
    out = []
    for lam in risk_aversions:
        traj = optimal_trajectory(problem, lam)
        out.append((float(lam), expected_cost(problem, traj), math.sqrt(cost_variance(problem, traj))))
    return out


if __name__ == "__main__":
    # The worked example from the paper: one million shares at $50, 30% annual vol,
    # liquidate over five days in ten half day intervals.
    price = 50.0
    daily_vol = 0.30 / math.sqrt(250) * price
    prob = ExecutionProblem(
        shares=1_000_000,
        horizon=5,
        periods=10,
        sigma=daily_vol,
        eta=2.5e-6,
        gamma=2.5e-7,
        epsilon=0.0625,
    )

    print(f"{'lambda':>10}{'E[cost] $':>14}{'Std $':>12}{'half life d':>13}")
    for lam, cost, std in efficient_frontier(prob, [0.0, 1e-7, 1e-6, 1e-5]):
        k = urgency(prob, lam)
        half = f"{1 / k:.2f}" if k > 0 else "inf"
        print(f"{lam:>10.0e}{cost:>14,.0f}{std:>12,.0f}{half:>13}")

    print("\nShares held at each step, lambda = 1e-6:")
    print(np.round(optimal_trajectory(prob, 1e-6)).astype(int))
    print("Shares held at each step, TWAP:")
    print(np.round(twap_trajectory(prob)).astype(int))
