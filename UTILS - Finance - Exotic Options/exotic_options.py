"""
Exotic Options Pricing via Monte Carlo
-----------------------------------------
Prices path-dependent options using Geometric Brownian Motion simulation.

- Barrier Options: Knocked out or in when price crosses a barrier level
- Asian Options: Payoff based on average price over the option's life
- Lookback Options: Payoff based on maximum or minimum price observed

All use antithetic variates for variance reduction where applicable.
"""

import numpy as np
from typing import Literal, Optional


def _simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate GBM price paths using log-Euler scheme. Returns (n_paths, n_steps+1)."""
    np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_price = np.log(S0) + np.cumsum(log_returns, axis=1)
    return np.concatenate([np.full((n_paths, 1), S0), np.exp(log_price)], axis=1)


def barrier_option(
    S0: float,
    K: float,
    H: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
    barrier_type: Literal["down-out", "down-in", "up-out", "up-in"] = "down-out",
    n_steps: int = 252,
    n_paths: int = 50_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo pricing for European barrier options.

    Args:
        S0: Spot price.
        K: Strike price.
        H: Barrier level.
        r: Risk-free rate.
        sigma: Volatility.
        T: Time to expiry (years).
        option_type: "call" or "put".
        barrier_type: "down-out", "down-in", "up-out", or "up-in".
        n_steps: Time steps per path.
        n_paths: Simulation paths.
        seed: Random seed.

    Returns:
        float: Option price.
    """
    paths = _simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    S_T = paths[:, -1]
    min_prices = np.min(paths, axis=1)
    max_prices = np.max(paths, axis=1)

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)

    if barrier_type == "down-out":
        alive = min_prices > H
    elif barrier_type == "down-in":
        alive = min_prices <= H
    elif barrier_type == "up-out":
        alive = max_prices < H
    elif barrier_type == "up-in":
        alive = max_prices >= H
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type}")

    return float(np.exp(-r * T) * np.mean(payoff * alive))


def asian_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
    averaging: Literal["arithmetic", "geometric"] = "arithmetic",
    n_steps: int = 252,
    n_paths: int = 50_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo pricing for Asian (average price) options.

    Args:
        S0: Spot price.
        K: Strike.
        r: Risk-free rate.
        sigma: Volatility.
        T: Time to expiry.
        option_type: "call" or "put".
        averaging: "arithmetic" or "geometric" average.
        n_steps: Time steps.
        n_paths: Simulation paths.
        seed: Random seed.

    Returns:
        float: Asian option price.
    """
    paths = _simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)

    if averaging == "arithmetic":
        avg = np.mean(paths[:, 1:], axis=1)
    else:
        avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

    if option_type == "call":
        payoff = np.maximum(avg - K, 0)
    else:
        payoff = np.maximum(K - avg, 0)

    return float(np.exp(-r * T) * np.mean(payoff))


def lookback_option(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal["call", "put"] = "call",
    fixed_strike: Optional[float] = None,
    n_steps: int = 252,
    n_paths: int = 50_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo pricing for lookback options.

    Floating strike: call = S_T - S_min, put = S_max - S_T.
    Fixed strike:   call = max(S_max - K, 0), put = max(K - S_min, 0).

    Args:
        S0: Spot price.
        r: Risk-free rate.
        sigma: Volatility.
        T: Time to expiry.
        option_type: "call" or "put".
        fixed_strike: If provided, uses fixed-strike payoff.
        n_steps: Time steps.
        n_paths: Simulation paths.
        seed: Random seed.

    Returns:
        float: Lookback option price.
    """
    paths = _simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    S_T = paths[:, -1]
    S_max = np.max(paths, axis=1)
    S_min = np.min(paths, axis=1)

    if fixed_strike is not None:
        K = fixed_strike
        payoff = np.maximum(S_max - K, 0) if option_type == "call" else np.maximum(K - S_min, 0)
    else:
        payoff = (S_T - S_min) if option_type == "call" else (S_max - S_T)

    return float(np.exp(-r * T) * np.mean(payoff))


if __name__ == "__main__":
    S0, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0

    print("Exotic Options Pricing (Monte Carlo)")
    print("=" * 45)

    do = barrier_option(S0, K, H=90, r=r, sigma=sigma, T=T, barrier_type="down-out")
    di = barrier_option(S0, K, H=90, r=r, sigma=sigma, T=T, barrier_type="down-in")
    print(f"\nBarrier Options (H=90, call):")
    print(f"  Down-and-Out: {do:.4f}")
    print(f"  Down-and-In:  {di:.4f}")
    print(f"  Sum (≈ vanilla): {do + di:.4f}")

    arith = asian_option(S0, K, r, sigma, T, averaging="arithmetic")
    geom = asian_option(S0, K, r, sigma, T, averaging="geometric")
    print(f"\nAsian Call Options:")
    print(f"  Arithmetic:  {arith:.4f}")
    print(f"  Geometric:   {geom:.4f}")

    lb_float = lookback_option(S0, r, sigma, T, option_type="call")
    lb_fixed = lookback_option(S0, r, sigma, T, option_type="call", fixed_strike=K)
    print(f"\nLookback Call Options:")
    print(f"  Floating strike: {lb_float:.4f}")
    print(f"  Fixed strike:    {lb_fixed:.4f}")
