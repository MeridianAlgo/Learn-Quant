"""
Binomial Tree Option Pricing (Cox-Ross-Rubinstein)
--------------------------------------------------
Black-Scholes gives a closed-form price for a European option, but it cannot
price an *American* option — one you may exercise early — and it hides the
mechanics behind a formula. The **binomial tree** does neither: it models the
underlying as a sequence of up/down moves over ``n`` discrete steps, then walks
backward from expiry discounting expected payoffs. It prices American options
naturally (just check early exercise at every node) and, as ``n`` grows, its
European price converges to Black-Scholes.

The Cox-Ross-Rubinstein (CRR) parameterisation picks the up/down factors so the
tree's volatility matches the underlying's:

    u = exp(sigma * sqrt(dt))      (up factor)
    d = 1 / u                      (down factor)
    p = (exp((r - q) * dt) - d) / (u - d)   (risk-neutral up probability)

with ``dt = T / n``. Pricing is then pure backward induction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def crr_parameters(sigma: float, T: float, n: int, r: float = 0.0, q: float = 0.0) -> Tuple[float, float, float]:
    """Return ``(u, d, p)`` for an ``n``-step CRR tree.

    ``u``/``d`` are the up/down multipliers and ``p`` is the risk-neutral
    probability of an up move. A no-arbitrage tree requires ``d < e^((r-q)dt) < u``;
    if that fails the inputs are inconsistent (e.g. ``n`` too small for ``sigma``).
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("risk-neutral probability outside [0, 1]; check inputs")
    return float(u), float(d), float(p)


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int = 200,
    option: str = "call",
    american: bool = False,
    q: float = 0.0,
) -> float:
    """Price a vanilla option on a CRR binomial tree.

    Args:
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiry in years.
        r: Continuously-compounded risk-free rate.
        sigma: Annualised volatility of returns.
        n: Number of time steps (more steps -> more accurate, slower).
        option: ``"call"`` or ``"put"``.
        american: If True, allow early exercise at every node.
        q: Continuous dividend yield.

    Returns:
        The option's present value.
    """
    option = option.lower()
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    u, d, p = crr_parameters(sigma, T, n, r, q)
    dt = T / n
    disc = np.exp(-r * dt)
    sign = 1.0 if option == "call" else -1.0

    # Terminal underlying prices: j up-moves, (n - j) down-moves.
    j = np.arange(n + 1)
    spot = S * u**j * d ** (n - j)
    values = np.maximum(sign * (spot - K), 0.0)

    # Walk backward through the tree, collapsing one column per step.
    for step in range(n - 1, -1, -1):
        spot = spot[: step + 1] / d  # underlying one step earlier
        values = disc * (p * values[1 : step + 2] + (1.0 - p) * values[: step + 1])
        if american:
            values = np.maximum(values, sign * (spot - K))
    return float(values[0])


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option: str = "call",
    american: bool = False,
    q: float = 0.0,
    n: int = 200,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Back out the volatility that reproduces an observed *price* (bisection).

    Bisection is slower than Newton's method but robust — it cannot diverge —
    which suits a tree price that has no clean vega in closed form.
    """
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = binomial_price(S, K, T, r, mid, n, option, american, q) - price
        if abs(diff) < tol:
            return mid
        # Price rises monotonically with volatility.
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("Binomial Tree Option Pricing")
    print("=" * 40)

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    euro_call = binomial_price(S, K, T, r, sigma, n=500, option="call")
    euro_put = binomial_price(S, K, T, r, sigma, n=500, option="put")
    amer_put = binomial_price(S, K, T, r, sigma, n=500, option="put", american=True)

    print(f"Spot {S}, strike {K}, T {T}y, r {r:.0%}, vol {sigma:.0%}\n")
    print(f"European call : {euro_call:.4f}")
    print(f"European put  : {euro_put:.4f}")
    print(f"American put  : {amer_put:.4f}  (>= European put: early exercise premium)")

    print("\nConvergence to Black-Scholes (European call) as steps grow:")
    for steps in (5, 25, 100, 500, 2000):
        price = binomial_price(S, K, T, r, sigma, n=steps, option="call")
        print(f"  n={steps:5d}: {price:.5f}")

    # Black-Scholes closed form for reference.
    from math import erf, exp, log, sqrt

    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    bs_call = S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    print(f"  Black-Scholes: {bs_call:.5f}")

    iv = implied_volatility(euro_call, S, K, T, r, option="call", n=500)
    print(f"\nImplied vol recovered from the call price: {iv:.4f}  (input was {sigma})")
