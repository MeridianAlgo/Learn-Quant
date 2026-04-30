"""
Short Rate Models: Vasicek and CIR
-------------------------------------
Models for simulating the short rate and pricing zero-coupon bonds.

- Vasicek (1977): dr = kappa*(theta - r)*dt + sigma*dW
  Mean-reverting; rates can go negative.

- Cox-Ingersoll-Ross (CIR, 1985): dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
  Mean-reverting; rates stay non-negative (when 2*kappa*theta >= sigma^2).

Both provide closed-form zero-coupon bond pricing formulas.
"""

import numpy as np
from typing import Optional


def vasicek_simulate(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate Vasicek short rate paths via Euler-Maruyama discretization.

    Args:
        r0: Initial short rate.
        kappa: Mean reversion speed.
        theta: Long-run mean rate.
        sigma: Volatility.
        T: Time horizon (years).
        n_steps: Number of time steps.
        n_paths: Number of simulation paths.
        seed: Random seed.

    Returns:
        np.ndarray: Shape (n_paths, n_steps+1) of rate paths.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0

    for t in range(n_steps):
        dW = np.random.randn(n_paths) * np.sqrt(dt)
        dr = kappa * (theta - paths[:, t]) * dt + sigma * dW
        paths[:, t + 1] = paths[:, t] + dr

    return paths


def vasicek_bond_price(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """
    Closed-form Vasicek zero-coupon bond price P(0, T).

    Returns:
        float: Bond price (discount factor from 0 to T).
    """
    B = (1 - np.exp(-kappa * T)) / kappa
    A_exp = np.exp(
        (theta - sigma**2 / (2 * kappa**2)) * (B - T)
        - sigma**2 * B**2 / (4 * kappa)
    )
    return float(A_exp * np.exp(-B * r0))


def vasicek_yield(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """
    Vasicek continuously compounded yield: y(0,T) = -ln(P(0,T)) / T.

    Returns:
        float: Zero-coupon yield for maturity T.
    """
    price = vasicek_bond_price(r0, kappa, theta, sigma, T)
    return float(-np.log(price) / T)


def cir_simulate(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate CIR short rate paths. Rates clamped at 0 (non-negativity).

    Args:
        r0: Initial short rate.
        kappa: Mean reversion speed.
        theta: Long-run mean.
        sigma: Volatility.
        T: Time horizon.
        n_steps: Time steps.
        n_paths: Number of paths.
        seed: Random seed.

    Returns:
        np.ndarray: Shape (n_paths, n_steps+1).
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0

    for t in range(n_steps):
        r_pos = np.maximum(paths[:, t], 0)
        dW = np.random.randn(n_paths) * np.sqrt(dt)
        dr = kappa * (theta - r_pos) * dt + sigma * np.sqrt(r_pos) * dW
        paths[:, t + 1] = np.maximum(paths[:, t] + dr, 0)

    return paths


def cir_bond_price(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """
    Closed-form CIR zero-coupon bond price P(0, T).

    Returns:
        float: Bond price.
    """
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    exp_gT = np.exp(gamma * T)

    denom = (gamma + kappa) * (exp_gT - 1) + 2 * gamma
    B = 2 * (exp_gT - 1) / denom
    A = (
        2 * gamma * np.exp((kappa + gamma) * T / 2) / denom
    ) ** (2 * kappa * theta / sigma**2)

    return float(A * np.exp(-B * r0))


def cir_yield(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """
    CIR continuously compounded yield for maturity T.

    Returns:
        float: Zero-coupon yield.
    """
    price = cir_bond_price(r0, kappa, theta, sigma, T)
    return float(-np.log(price) / T)


def term_structure(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    maturities: list,
    model: str = "vasicek",
) -> dict:
    """
    Compute the full term structure (yield curve) for given maturities.

    Args:
        r0, kappa, theta, sigma: Model parameters.
        maturities: List of maturities in years.
        model: "vasicek" or "cir".

    Returns:
        dict: Maturity → yield.
    """
    yield_fn = vasicek_yield if model.lower() == "vasicek" else cir_yield
    return {T: yield_fn(r0, kappa, theta, sigma, T) for T in maturities}


if __name__ == "__main__":
    r0, kappa, theta, sigma = 0.03, 0.30, 0.05, 0.01
    maturities = [0.25, 0.5, 1, 2, 5, 10, 20, 30]

    print("Interest Rate Models — Term Structure")
    print("=" * 45)
    print(f"Parameters: r0={r0:.2%}, kappa={kappa}, theta={theta:.2%}, sigma={sigma:.2%}")
    print(f"\n{'Maturity':>8} | {'Vasicek Yield':>14} | {'CIR Yield':>10}")
    print("-" * 40)
    for T in maturities:
        vy = vasicek_yield(r0, kappa, theta, sigma, T)
        cy = cir_yield(r0, kappa, theta, sigma, T)
        print(f"  {T:5.2f}y | {vy:14.4%} | {cy:10.4%}")

    paths = vasicek_simulate(r0, kappa, theta, sigma, T=1.0, n_steps=252, n_paths=3, seed=42)
    print(f"\nVasicek 3-path simulation final rates: {paths[:, -1]}")
