"""
Implied Volatility & the Volatility Surface
-------------------------------------------
Black-Scholes takes volatility as an input and returns a price. The market does
the opposite: it quotes a price, and we invert the formula to find the
volatility that reproduces it — the **implied volatility (IV)**.

Plot IV against strike and maturity and you get the **volatility surface**: the
"smile"/"skew" across strikes and the term structure across maturities. A flat
surface would mean Black-Scholes is literally true; the fact that it is never
flat is the single most important empirical fact in option pricing.

This module prices European options, inverts for IV with a robust
Newton + bisection solver, and builds / interpolates a surface from a grid of
quotes — all with NumPy only.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[list, np.ndarray]
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option: str = "call", q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European option (with dividend yield q)."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return float(intrinsic)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Vega — sensitivity of price to volatility (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Invert Black-Scholes for implied volatility.

    Newton's method (fast, uses vega) with a bisection safety net so it never
    diverges on near-the-money or deep options.

    Returns ``nan`` if the quoted price violates no-arbitrage bounds.
    """
    intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
    upper = S if option == "call" else K
    if price < intrinsic - 1e-12 or price > upper + 1e-12:
        return float("nan")

    lo, hi = 1e-6, 5.0
    sigma = 0.2  # sensible starting guess
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option, q) - price
        if abs(diff) < tol:
            return float(sigma)
        v = bs_vega(S, K, T, r, sigma, q)
        # Maintain a bracket for the bisection fallback.
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        if v > 1e-10:
            step = diff / v
            sigma_newton = sigma - step
            if lo < sigma_newton < hi:
                sigma = sigma_newton
                continue
        sigma = 0.5 * (lo + hi)  # bisection step
    return float(sigma)


class VolSurface:
    """A discrete implied-volatility surface with bilinear interpolation.

    Built from a grid of strikes, maturities and option *prices* (it inverts
    each to IV on construction). Query any ``(K, T)`` with :meth:`iv`.
    """

    def __init__(self, S: float, r: float, q: float = 0.0):
        self.S = S
        self.r = r
        self.q = q
        self.strikes: np.ndarray | None = None
        self.maturities: np.ndarray | None = None
        self.iv_grid: np.ndarray | None = None

    def fit(self, strikes: ArrayLike, maturities: ArrayLike, price_grid: ArrayLike, option: str = "call") -> VolSurface:
        """Build the surface from a ``len(maturities) x len(strikes)`` price grid."""
        self.strikes = np.asarray(strikes, dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        prices = np.asarray(price_grid, dtype=float)
        if prices.shape != (len(self.maturities), len(self.strikes)):
            raise ValueError("price_grid must have shape (len(maturities), len(strikes))")
        iv = np.empty_like(prices)
        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.strikes):
                iv[i, j] = implied_vol(prices[i, j], self.S, K, T, self.r, option, self.q)
        self.iv_grid = iv
        return self

    def iv(self, K: float, T: float) -> float:
        """Interpolated implied volatility at strike ``K`` and maturity ``T``."""
        if self.iv_grid is None:
            raise RuntimeError("call fit() before querying the surface")
        ks, ts, grid = self.strikes, self.maturities, self.iv_grid
        K = float(np.clip(K, ks[0], ks[-1]))
        T = float(np.clip(T, ts[0], ts[-1]))
        j = int(np.clip(np.searchsorted(ks, K) - 1, 0, len(ks) - 2))
        i = int(np.clip(np.searchsorted(ts, T) - 1, 0, len(ts) - 2))
        # Bilinear weights.
        tk = (K - ks[j]) / (ks[j + 1] - ks[j]) if ks[j + 1] != ks[j] else 0.0
        tt = (T - ts[i]) / (ts[i + 1] - ts[i]) if ts[i + 1] != ts[i] else 0.0
        v00, v01 = grid[i, j], grid[i, j + 1]
        v10, v11 = grid[i + 1, j], grid[i + 1, j + 1]
        top = v00 * (1 - tk) + v01 * tk
        bot = v10 * (1 - tk) + v11 * tk
        return float(top * (1 - tt) + bot * tt)

    def smile(self, T: float) -> np.ndarray:
        """The IV smile (one IV per strike) at maturity ``T``."""
        return np.array([self.iv(K, T) for K in self.strikes])


if __name__ == "__main__":
    S, r = 100.0, 0.02
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    maturities = np.array([0.25, 0.5, 1.0], dtype=float)

    # Synthesise a market with a skew (lower strikes richer) and term structure.
    def true_iv(K, T):
        moneyness = math.log(K / S)
        return 0.20 - 0.35 * moneyness + 0.05 * math.sqrt(T)

    price_grid = np.array([[bs_price(S, K, T, r, true_iv(K, T), "call") for K in strikes] for T in maturities])

    surf = VolSurface(S, r).fit(strikes, maturities, price_grid, option="call")

    print("Recovered Implied Volatility Surface (calls)")
    print("=" * 52)
    header = "  T \\ K " + "".join(f"{int(k):>9}" for k in strikes)
    print(header)
    for i, T in enumerate(maturities):
        row = "".join(f"{surf.iv_grid[i, j]:>9.3f}" for j in range(len(strikes)))
        print(f"{T:>6.2f} {row}")

    print("\nInterpolated IV at K=105, T=0.4:")
    print(f"  {surf.iv(105, 0.4):.4f}  (true {true_iv(105, 0.4):.4f})")
    print("\nThe downward slope across strikes is the equity 'volatility skew'.")
