"""
Kernel Density Estimation
-------------------------
A histogram answers "what does this distribution look like" with a picture
that changes shape depending on where you put the bin edges. A kernel density
estimate answers the same question without bins. It drops a small smooth bump
on every observation and adds the bumps up.

The result is a smooth curve you can evaluate anywhere, integrate to get
probabilities, and sample from. For returns that is useful, because the
normal curve is a poor fit in the tails and an empirical histogram is too
jagged to read a quantile off cleanly.

The one decision that matters is the width of the bump, the bandwidth. Too
narrow and you get a spike on every point. Too wide and every distribution
looks normal. Silverman's rule is a sensible default and the reasoning behind
it is in the README.
"""

from __future__ import annotations

import math

import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _as_sample(data) -> np.ndarray:
    x = np.asarray(data, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("need at least two observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("data contains NaN or infinite values")
    return x


def gaussian_kernel(u):
    """Standard normal density, the smooth bump placed on every point."""
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u * u) / SQRT_2PI


def epanechnikov_kernel(u):
    """Parabolic bump that is exactly zero beyond one bandwidth."""
    u = np.asarray(u, dtype=float)
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)


KERNELS = {"gaussian": gaussian_kernel, "epanechnikov": epanechnikov_kernel}


def silverman_bandwidth(data) -> float:
    """Silverman's rule of thumb, robust to a few outliers.

    ``h = 0.9 * min(std, IQR / 1.34) * n ** (-1/5)``. Using the smaller of the
    two spreads stops one huge return from inflating the bandwidth and
    flattening the whole curve.
    """
    x = _as_sample(data)
    std = x.std(ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    spread = min(std, (q75 - q25) / 1.34) if q75 > q25 else std
    if spread <= 0:
        raise ValueError("data has no spread, bandwidth is undefined")
    return 0.9 * spread * x.size ** (-0.2)


def scott_bandwidth(data) -> float:
    """Scott's rule, ``1.06 * std * n ** (-1/5)``, optimal if the data is normal."""
    x = _as_sample(data)
    std = x.std(ddof=1)
    if std <= 0:
        raise ValueError("data has no spread, bandwidth is undefined")
    return 1.06 * std * x.size ** (-0.2)


def kde(data, points, bandwidth: float | None = None, kernel: str = "gaussian") -> np.ndarray:
    """Evaluate the density estimate at ``points``.

    ``f(x) = (1 / (n h)) * sum K((x - x_i) / h)``.
    """
    x = _as_sample(data)
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}, choose from {sorted(KERNELS)}")
    h = silverman_bandwidth(x) if bandwidth is None else float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be positive")
    grid = np.atleast_1d(np.asarray(points, dtype=float))
    u = (grid[:, None] - x[None, :]) / h
    return KERNELS[kernel](u).sum(axis=1) / (x.size * h)


def kde_cdf(data, points, bandwidth: float | None = None) -> np.ndarray:
    """Cumulative probability under the Gaussian KDE.

    Each bump integrates to a normal CDF, so the smoothed CDF is the average
    of ``Phi((x - x_i) / h)`` over the sample.
    """
    x = _as_sample(data)
    h = silverman_bandwidth(x) if bandwidth is None else float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be positive")
    grid = np.atleast_1d(np.asarray(points, dtype=float))
    z = (grid[:, None] - x[None, :]) / (h * math.sqrt(2.0))
    erf = np.vectorize(math.erf)
    return (0.5 * (1.0 + erf(z))).mean(axis=1)


def kde_quantile(data, prob: float, bandwidth: float | None = None, tol: float = 1e-10) -> float:
    """Invert the smoothed CDF by bisection to get a quantile."""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be strictly between 0 and 1")
    x = _as_sample(data)
    h = silverman_bandwidth(x) if bandwidth is None else float(bandwidth)
    lo, hi = x.min() - 10 * h, x.max() + 10 * h
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kde_cdf(x, mid, h)[0] < prob:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def kde_var(returns, confidence: float = 0.95, bandwidth: float | None = None) -> float:
    """Value at Risk read off the smoothed distribution, reported as a positive loss."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    return -kde_quantile(returns, 1.0 - confidence, bandwidth)


def sample_kde(data, size: int, bandwidth: float | None = None, seed: int | None = None) -> np.ndarray:
    """Draw from the Gaussian KDE, also known as the smoothed bootstrap.

    Pick an observation at random, then add normal noise with standard
    deviation equal to the bandwidth. Unlike the plain bootstrap this can
    produce values that never appeared in the sample.
    """
    x = _as_sample(data)
    if size < 1:
        raise ValueError("size must be at least 1")
    h = silverman_bandwidth(x) if bandwidth is None else float(bandwidth)
    rng = np.random.default_rng(seed)
    return rng.choice(x, size=size, replace=True) + rng.normal(0.0, h, size=size)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    # Student t with 4 degrees of freedom, scaled to look like daily returns.
    rets = 0.01 * rng.standard_t(4, size=1500)

    h_s = silverman_bandwidth(rets)
    h_scott = scott_bandwidth(rets)
    print(f"Silverman bandwidth: {h_s:.5f}")
    print(f"Scott bandwidth:     {h_scott:.5f}")

    grid = np.linspace(-0.06, 0.06, 7)
    dens = kde(rets, grid)
    normal = np.exp(-0.5 * ((grid - rets.mean()) / rets.std()) ** 2) / (rets.std() * SQRT_2PI)
    print(f"\n{'x':>8}{'KDE':>10}{'Normal':>10}")
    for g, d, nrm in zip(grid, dens, normal):
        print(f"{g:>8.3f}{d:>10.3f}{nrm:>10.3f}")

    for conf in (0.95, 0.99):
        hist = -np.percentile(rets, 100 * (1 - conf))
        print(f"\n{conf:.0%} VaR  historical {hist:.4f}  KDE {kde_var(rets, conf):.4f}")

    draws = sample_kde(rets, 5, seed=1)
    print("\nFive smoothed bootstrap draws:", np.round(draws, 4))
