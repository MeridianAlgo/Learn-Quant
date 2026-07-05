"""
Monte Carlo Integration
-----------------------
Some integrals have no tidy formula, and in high dimensions the grid methods
from numerical analysis collapse under their own cost. Monte Carlo sidesteps
both problems with an idea that sounds too simple to work. An integral is an
average, so estimate the average by sampling. Throw random points at the
function, take their mean, and the law of large numbers walks you toward the
answer at a rate that does not care how many dimensions the problem has.

The price is noise. The estimate carries a standard error that shrinks with
the square root of the sample count, so each extra digit of accuracy costs a
hundred times more samples. Variance reduction tricks like antithetic
sampling claw some of that back for free. The lesson closes with the flagship
financial application, pricing a European option by simulating where the
stock might end up, checked against the Black Scholes closed form.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def mc_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate the integral of f over [a, b] and its standard error.

    Uniform points are drawn on the interval, the function values are
    averaged, and the average is scaled by the interval width. The second
    number returned is the standard error of the estimate, the one sigma
    uncertainty, which shrinks like one over the square root of n.
    """
    if b <= a:
        raise ValueError("need b greater than a")
    if n < 2:
        raise ValueError("need at least two samples")
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    values = np.asarray(f(x), dtype=float)
    width = b - a
    estimate = width * float(values.mean())
    stderr = width * float(values.std(ddof=1)) / math.sqrt(n)
    return estimate, stderr


def antithetic_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate the same integral using antithetic pairs.

    Every uniform draw u is paired with its mirror image a + b - u. The two
    points are negatively correlated, so averaging each pair cancels part of
    the noise before it ever reaches the estimate. For a monotone function the
    variance drops substantially at zero extra sampling cost.
    """
    if b <= a:
        raise ValueError("need b greater than a")
    if n < 4:
        raise ValueError("need at least four samples")
    rng = np.random.default_rng(seed)
    u = rng.uniform(a, b, size=n // 2)
    pair_means = (np.asarray(f(u), dtype=float) + np.asarray(f(a + b - u), dtype=float)) / 2
    width = b - a
    estimate = width * float(pair_means.mean())
    stderr = width * float(pair_means.std(ddof=1)) / math.sqrt(len(pair_means))
    return estimate, stderr


def mc_european_call(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    n: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Price a European call by simulating the terminal stock price.

    Under the risk neutral measure the stock ends at spot times a lognormal
    factor. Each simulated ending gives one payoff, the discounted average of
    the payoffs is the price, and the spread of the payoffs gives the standard
    error. This is the workhorse pattern behind pricing anything too exotic
    for a closed form.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    terminal = spot * np.exp((rate - 0.5 * sigma**2) * maturity + sigma * math.sqrt(maturity) * z)
    payoffs = np.exp(-rate * maturity) * np.clip(terminal - strike, 0, None)
    return float(payoffs.mean()), float(payoffs.std(ddof=1)) / math.sqrt(n)


def black_scholes_call(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    """Return the Black Scholes call price, the exact answer to check against."""
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * math.sqrt(maturity))
    d2 = d1 - sigma * math.sqrt(maturity)
    norm_cdf = lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2)))  # noqa: E731
    return spot * norm_cdf(d1) - strike * math.exp(-rate * maturity) * norm_cdf(d2)


if __name__ == "__main__":
    print("Monte Carlo Integration")
    print("=" * 40)

    print("\nIntegral of sin(x) from 0 to pi, exact answer 2")
    est, err = mc_integrate(np.sin, 0, math.pi, n=50_000, seed=1)
    print(f"  plain estimate      {est:.4f}  +/- {err:.4f}")

    print("\nAntithetic pairs cut the noise on a monotone function")
    print("Integral of exp(x) from 0 to 1, exact answer e - 1 = 1.7183")
    est_p, err_p = mc_integrate(np.exp, 0, 1, n=50_000, seed=1)
    est_a, err_a = antithetic_integrate(np.exp, 0, 1, n=50_000, seed=1)
    print(f"  plain estimate      {est_p:.4f}  +/- {err_p:.5f}")
    print(f"  antithetic estimate {est_a:.4f}  +/- {err_a:.5f}")

    print("\nPricing a European call, spot 100, strike 105, r 5%, vol 20%, 1 year")
    exact = black_scholes_call(100, 105, 0.05, 0.2, 1.0)
    mc_price, mc_err = mc_european_call(100, 105, 0.05, 0.2, 1.0, n=200_000, seed=2)
    print(f"  Black Scholes  {exact:.4f}")
    print(f"  Monte Carlo    {mc_price:.4f}  +/- {mc_err:.4f}")

    print("\nThe error shrinks like one over the square root of n")
    for n in (1_000, 10_000, 100_000):
        _, err = mc_integrate(np.sin, 0, math.pi, n=n, seed=3)
        print(f"  n = {n:>7,}   standard error {err:.5f}")
