"""
Bootstrap Resampling for Finance
---------------------------------
The bootstrap estimates the sampling distribution of *any* statistic by
resampling the observed data with replacement, instead of relying on a
parametric formula (which often assumes normality that financial returns
violate). It answers questions like:

- "My backtest has a Sharpe ratio of 1.4 — what is the 95% confidence interval?"
- "Is this strategy's mean return significantly different from zero?"
- "How uncertain is my estimate of maximum drawdown?"

Three flavours are implemented:
- i.i.d. bootstrap        — resample individual observations (assumes no serial
                            dependence; fine for already-uncorrelated returns).
- block bootstrap         — resample contiguous blocks to preserve short-term
                            autocorrelation and volatility clustering.
- stationary bootstrap    — Politis & Romano (1994): blocks of random geometric
                            length, which keeps the resampled series stationary.

Because financial returns exhibit autocorrelation and volatility clustering,
block-based methods usually give more honest confidence intervals than the
naive i.i.d. bootstrap.
"""

from typing import Callable

import numpy as np


def iid_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = None,
) -> np.ndarray:
    """
    Classic i.i.d. bootstrap: resample individual observations with replacement.

    Args:
        data: 1-D array of observations (e.g. daily returns).
        statistic: Function mapping a sample to a scalar (e.g. np.mean, sharpe).
        n_boot: Number of bootstrap replications.
        seed: Optional RNG seed for reproducibility.

    Returns:
        np.ndarray: length-n_boot array of the statistic across resamples.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        # Draw n indices with replacement → one synthetic dataset.
        idx = rng.integers(0, n, size=n)
        estimates[i] = statistic(data[idx])
    return estimates


def block_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    block_size: int = 20,
    n_boot: int = 1000,
    seed: int = None,
) -> np.ndarray:
    """
    Moving-block bootstrap: resample fixed-length contiguous blocks.

    Preserves autocorrelation up to roughly `block_size` lags, so it is the
    appropriate choice for serially dependent series like returns or volatility.

    Args:
        data: 1-D array of observations.
        statistic: Function mapping a sample to a scalar.
        block_size: Length of each contiguous block.
        n_boot: Number of bootstrap replications.
        seed: Optional RNG seed.

    Returns:
        np.ndarray: length-n_boot array of the statistic across resamples.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    n_blocks = int(np.ceil(n / block_size))
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        # Pick random block start points, glue blocks together, trim to length n.
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([data[s : s + block_size] for s in starts])[:n]
        estimates[i] = statistic(sample)
    return estimates


def stationary_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    expected_block: float = 20.0,
    n_boot: int = 1000,
    seed: int = None,
) -> np.ndarray:
    """
    Stationary bootstrap (Politis & Romano, 1994).

    Like the block bootstrap but block lengths are random (geometric with mean
    `expected_block`), and the series wraps around circularly. This guarantees
    the resampled series is stationary, removing the block-boundary artefacts of
    the fixed-block method.

    Args:
        data: 1-D array of observations.
        statistic: Function mapping a sample to a scalar.
        expected_block: Mean block length; p = 1 / expected_block.
        n_boot: Number of bootstrap replications.
        seed: Optional RNG seed.

    Returns:
        np.ndarray: length-n_boot array of the statistic across resamples.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    p = 1.0 / expected_block  # probability of starting a fresh block each step
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        sample = np.empty(n)
        idx = rng.integers(0, n)  # random starting index
        for t in range(n):
            if rng.random() < p:
                idx = rng.integers(0, n)  # jump to a new random block
            sample[t] = data[idx]
            idx = (idx + 1) % n  # advance within the block, wrapping circularly
        estimates[b] = statistic(sample)
    return estimates


def confidence_interval(estimates: np.ndarray, alpha: float = 0.05) -> tuple:
    """
    Percentile confidence interval from bootstrap estimates.

    Args:
        estimates: Array of bootstrapped statistic values.
        alpha: Significance level (0.05 → 95% interval).

    Returns:
        tuple: (lower, upper) bounds of the (1 - alpha) interval.
    """
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


if __name__ == "__main__":
    print("Bootstrap Resampling — Sharpe Ratio Confidence Interval")
    print("=" * 56)

    rng = np.random.default_rng(7)
    # Simulate 2 years of daily returns with a modest positive drift.
    returns = rng.normal(0.0005, 0.012, 504)

    def annual_sharpe(x: np.ndarray) -> float:
        # Daily Sharpe annualised by sqrt(252).
        mu = x.mean()
        sd = x.std(ddof=1)
        return float(mu / sd * np.sqrt(252)) if sd > 0 else 0.0

    point = annual_sharpe(returns)
    print(f"\nPoint estimate Sharpe: {point:.3f}\n")

    for name, fn in [
        ("i.i.d.", iid_bootstrap),
        ("block (20d)", lambda d, s, **k: block_bootstrap(d, s, block_size=20, **k)),
        ("stationary", lambda d, s, **k: stationary_bootstrap(d, s, expected_block=20, **k)),
    ]:
        est = fn(returns, annual_sharpe, n_boot=2000, seed=1)
        lo, hi = confidence_interval(est)
        print(f"  {name:14s} 95% CI: [{lo:+.3f}, {hi:+.3f}]   mean {est.mean():+.3f}")
