"""
Omega Ratio
-----------
The Sharpe ratio compresses a whole return distribution into a mean and a
standard deviation, which throws away everything interesting about its shape.
The Omega ratio keeps the shape. Pick a threshold return you care about, then
compare all the probability weighted gains above it to all the probability
weighted losses below it. The result is a single number that uses every moment
of the distribution, skew and fat tails included, without ever estimating them.

An Omega of one means gains above the threshold exactly balance losses below
it. Above one the strategy earned more than it gave up relative to your bar.
Because the threshold is yours to choose, the same track record has a whole
Omega curve, and comparing two strategies across thresholds tells you where
each one earns its keep.
"""

from __future__ import annotations

import numpy as np


def excess_gains_and_losses(returns: list[float], threshold: float = 0.0) -> tuple[float, float]:
    """Split a return series into gains above and losses below a threshold.

    Returns the average excess gain and the average excess loss, both as
    positive numbers. These are the two halves the Omega ratio compares.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        raise ValueError("need at least one return")
    excess = r - threshold
    gains = float(np.clip(excess, 0, None).mean())
    losses = float(-np.clip(excess, None, 0).mean())
    return gains, losses


def omega_ratio(returns: list[float], threshold: float = 0.0) -> float:
    """Return the Omega ratio at the given per period threshold.

    This is the sum of returns above the threshold divided by the sum of
    shortfalls below it. A value above one means the strategy delivered more
    excess gain than excess pain relative to the bar you set. With no losses
    below the threshold at all the ratio is infinite, reported as positive
    infinity.
    """
    gains, losses = excess_gains_and_losses(returns, threshold)
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return gains / losses


def omega_curve(returns: list[float], thresholds: list[float]) -> list[tuple[float, float]]:
    """Return (threshold, omega) pairs across a range of thresholds.

    The curve always falls as the threshold rises, because raising the bar
    shrinks the gains and grows the shortfalls. Where two strategies' curves
    cross tells you which one suits an investor with that particular bar.
    """
    return [(float(t), omega_ratio(returns, t)) for t in thresholds]


def annual_threshold_to_periodic(annual: float, periods_per_year: int = 252) -> float:
    """Convert an annual threshold like 0.05 into a per period one.

    Omega works on per period returns, but people think in annual terms. This
    converts a yearly bar into the equivalent compounding per period rate so
    the two scales agree.
    """
    return (1 + annual) ** (1 / periods_per_year) - 1


if __name__ == "__main__":
    print("Omega Ratio")
    print("=" * 40)

    rng = np.random.default_rng(11)
    steady = rng.normal(0.0005, 0.008, size=500).tolist()
    lumpy = rng.standard_t(3, size=500) * 0.01 + 0.0005
    lumpy = lumpy.tolist()

    print("\nAt a threshold of zero")
    print(f"  steady strategy  omega {omega_ratio(steady):.2f}")
    print(f"  fat tailed one   omega {omega_ratio(lumpy):.2f}")

    bar = annual_threshold_to_periodic(0.05)
    print(f"\nAt an annual bar of five percent ({bar:.5f} per day)")
    print(f"  steady strategy  omega {omega_ratio(steady, bar):.2f}")
    print(f"  fat tailed one   omega {omega_ratio(lumpy, bar):.2f}")

    print("\nThe omega curve falls as the bar rises")
    for t, om in omega_curve(steady, [0.0, 0.0005, 0.001]):
        print(f"  threshold {t:.4f}  omega {om:.2f}")
