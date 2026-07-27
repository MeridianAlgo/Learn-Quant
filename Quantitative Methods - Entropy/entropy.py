"""
Entropy and Information
-----------------------
Entropy measures how surprised you should expect to be. A coin that always
lands heads carries no information because you already know the answer, and
its entropy is zero. A fair coin is the most uncertain a two sided bet can
be, and in base two its entropy is exactly one bit. Everything else sits
between those extremes.

The same idea answers two other questions that come up constantly. Kullback
Leibler divergence measures the cost of believing the wrong distribution, so
it is the natural way to score how badly a model's assumed distribution
misses the realised one. Mutual information measures how much knowing one
variable reduces uncertainty about another, and unlike correlation it does
not care whether the relationship is a straight line, which makes it useful
for spotting a feature whose link to returns is real but not linear.
"""

from __future__ import annotations

import numpy as np


def shannon_entropy(probabilities: list[float], base: float = 2.0) -> float:
    """Return the Shannon entropy of a probability distribution.

    Base two gives an answer in bits, the natural log gives nats. Zero
    probability outcomes contribute nothing, which is the usual convention
    since a thing that never happens carries no surprise.
    """
    p = np.asarray(probabilities, dtype=float)
    if np.any(p < 0):
        raise ValueError("probabilities cannot be negative")
    total = p.sum()
    if total <= 0:
        raise ValueError("probabilities must sum to something positive")
    p = p / total
    nonzero = p[p > 0]
    # The trailing addition turns a negative zero into a plain zero.
    return float(-np.sum(nonzero * np.log(nonzero)) / np.log(base)) + 0.0


def entropy_from_samples(samples: list, base: float = 2.0) -> float:
    """Return the entropy of the empirical distribution of some samples.

    The samples are counted into their distinct values and those counts
    become the probabilities. This is the version you reach for with a
    sequence of observed states rather than a known distribution.
    """
    if len(samples) == 0:
        raise ValueError("need at least one sample")
    _, counts = np.unique(np.asarray(samples), return_counts=True)
    return shannon_entropy(counts.tolist(), base)


def normalized_entropy(probabilities: list[float]) -> float:
    """Return entropy as a fraction of the maximum possible for its size.

    Dividing by the entropy of the uniform distribution over the same number
    of outcomes puts the answer on a zero to one scale, which makes
    distributions of different sizes comparable. A single outcome is defined
    as fully certain and scores zero.
    """
    n = np.asarray(probabilities, dtype=float).size
    if n <= 1:
        return 0.0
    return shannon_entropy(probabilities) / np.log2(n)


def kl_divergence(p: list[float], q: list[float], base: float = 2.0) -> float:
    """Return the Kullback Leibler divergence from `q` to `p`.

    This is the extra cost, in bits, of describing outcomes drawn from `p`
    using a code built for `q`. It is never negative and it is zero only when
    the two distributions match. It is not symmetric, so the divergence from
    `q` to `p` is a different number, and it is infinite when `q` assigns zero
    probability to something `p` considers possible.
    """
    p_array = np.asarray(p, dtype=float)
    q_array = np.asarray(q, dtype=float)
    if p_array.size != q_array.size:
        raise ValueError("distributions must have the same number of outcomes")
    if np.any(p_array < 0) or np.any(q_array < 0):
        raise ValueError("probabilities cannot be negative")
    p_array = p_array / p_array.sum()
    q_array = q_array / q_array.sum()
    mask = p_array > 0
    if np.any(q_array[mask] == 0):
        return float("inf")
    ratio = p_array[mask] / q_array[mask]
    return float(np.sum(p_array[mask] * np.log(ratio)) / np.log(base))


def mutual_information(x: list, y: list, base: float = 2.0) -> float:
    """Return the mutual information between two sequences of labels.

    Mutual information is the entropy of `x` plus the entropy of `y` minus
    the entropy of the pair. It is zero when the two are independent and it
    grows as one becomes more predictable from the other. Unlike correlation
    it detects relationships of any shape, including ones that curve back on
    themselves.
    """
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    if x_array.size != y_array.size:
        raise ValueError("x and y must be the same length")
    if x_array.size == 0:
        raise ValueError("need at least one observation")
    pairs = [f"{a}|{b}" for a, b in zip(x_array.tolist(), y_array.tolist())]
    joint = entropy_from_samples(pairs, base)
    return entropy_from_samples(x_array.tolist(), base) + entropy_from_samples(y_array.tolist(), base) - joint


def discretize(values: list[float], bins: int = 10) -> list[int]:
    """Return bin labels for continuous values using equal width buckets.

    Entropy is defined over discrete outcomes, so a continuous series such as
    a return stream has to be bucketed first. The choice of bin count matters
    and too many bins will manufacture information out of noise.
    """
    if bins < 1:
        raise ValueError("need at least one bin")
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("need at least one value")
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        return [0] * v.size
    edges = np.linspace(lo, hi, bins + 1)[1:-1]
    return np.digitize(v, edges).tolist()


if __name__ == "__main__":
    print("Entropy and Information")
    print("=" * 40)

    print("\nEntropy of some two outcome bets, in bits")
    for p in (0.5, 0.7, 0.9, 0.99, 1.0):
        print(f"  p = {p:<5} {shannon_entropy([p, 1 - p]):.4f}")

    print("\nA fair coin is the most uncertain, a certain outcome carries nothing.")

    print("\nKL divergence of a fair coin from biased models")
    for q in (0.5, 0.6, 0.8):
        print(f"  model p = {q:<4} {kl_divergence([0.5, 0.5], [q, 1 - q]):.4f}")

    rng = np.random.default_rng(7)
    market = rng.normal(0, 0.01, size=4000)
    tied = np.sign(market) * np.abs(rng.normal(0, 0.01, size=4000))
    independent = rng.normal(0, 0.01, size=4000)

    m = discretize(market.tolist(), bins=8)
    print("\nMutual information with the market, in bits")
    print(f"  same sign series  {mutual_information(m, discretize(tied.tolist(), 8)):.4f}")
    print(f"  independent noise {mutual_information(m, discretize(independent.tolist(), 8)):.4f}")

    print("\nThe sign linked series shares information even though it is not a straight line.")
