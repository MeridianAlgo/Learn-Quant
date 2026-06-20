"""
Markov Chains
-------------
A Markov chain models a system that hops between a finite set of *states*, where
the next state depends only on the current one — not on the whole history. That
"memoryless" assumption is crude but powerful: in quant work it is the backbone
of regime models (bull / bear / sideways markets), credit-rating migration, and
simple weather-style models of volatility clustering.

Everything lives in one object: the **transition matrix** ``P``, where
``P[i, j]`` is the probability of moving from state ``i`` to state ``j`` in one
step. Each row sums to 1 (it is a probability distribution over where you go
next). From ``P`` alone you can read off:

- where you'll be in ``n`` steps (matrix power ``P^n``),
- the long-run fraction of time in each state (the **stationary distribution**),
- and you can simulate sample paths.
"""

from __future__ import annotations

import numpy as np


def normalize_rows(matrix) -> np.ndarray:
    """Turn a matrix of non-negative counts/weights into a row-stochastic one.

    Handy for estimating ``P`` from observed transition counts: feed in the
    count matrix and each row is divided by its total. A row of all zeros (a
    state never left) is mapped to a uniform distribution so ``P`` stays valid.
    """
    matrix = np.asarray(matrix, dtype=float)
    if np.any(matrix < 0):
        raise ValueError("transition weights must be non-negative")
    sums = matrix.sum(axis=1, keepdims=True)
    out = np.where(sums == 0, 1.0 / matrix.shape[1], matrix / np.where(sums == 0, 1.0, sums))
    return out


def is_stochastic(P, tol: float = 1e-9) -> bool:
    """True if every row of *P* is a probability distribution (non-negative, sums to 1)."""
    P = np.asarray(P, dtype=float)
    return bool(np.all(P >= -tol) and np.allclose(P.sum(axis=1), 1.0, atol=tol))


def n_step(P, n: int) -> np.ndarray:
    """The ``n``-step transition matrix ``P^n``.

    ``(P^n)[i, j]`` is the probability of being in state ``j`` exactly ``n``
    steps after starting in ``i`` (Chapman-Kolmogorov).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.linalg.matrix_power(np.asarray(P, dtype=float), n)


def stationary_distribution(P) -> np.ndarray:
    """The long-run distribution ``pi`` satisfying ``pi P = pi``.

    Computed as the left eigenvector of ``P`` for eigenvalue 1, normalised to sum
    to 1. For an irreducible, aperiodic chain this is unique and equals the
    fraction of time spent in each state over a long path.
    """
    P = np.asarray(P, dtype=float)
    if not is_stochastic(P):
        raise ValueError("P must be row-stochastic")
    # Left eigenvectors of P are right eigenvectors of P^T.
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi)  # eigenvector sign is arbitrary
    return pi / pi.sum()


def simulate(P, start: int, steps: int, random_state: int | None = None) -> np.ndarray:
    """Generate a sample path of ``steps`` states starting from ``start``.

    Returns an integer array of length ``steps + 1`` (including the start state).
    """
    P = np.asarray(P, dtype=float)
    rng = np.random.default_rng(random_state)
    n_states = P.shape[0]
    path = np.empty(steps + 1, dtype=int)
    path[0] = start
    for t in range(steps):
        path[t + 1] = rng.choice(n_states, p=P[path[t]])
    return path


def expected_return_time(P, state: int) -> float:
    """Mean number of steps to return to *state*, ``1 / pi[state]``.

    A consequence of the stationary distribution: states visited often (large
    ``pi``) are returned to quickly.
    """
    pi = stationary_distribution(P)
    if pi[state] == 0:
        return float("inf")
    return float(1.0 / pi[state])


if __name__ == "__main__":
    print("Markov Chains")
    print("=" * 40)

    states = ["Bull", "Bear", "Flat"]
    # Daily regime transitions: markets are sticky, so the diagonal dominates.
    P = np.array(
        [
            [0.90, 0.03, 0.07],  # from Bull
            [0.05, 0.85, 0.10],  # from Bear
            [0.15, 0.10, 0.75],  # from Flat
        ]
    )
    print("Row-stochastic transition matrix:", is_stochastic(P))

    pi = stationary_distribution(P)
    print("\nLong-run regime mix (stationary distribution):")
    for s, p in zip(states, pi):
        print(f"  {s:5s}: {p:6.2%}")

    print("\nReturn times (avg days to revisit a regime):")
    for i, s in enumerate(states):
        print(f"  {s:5s}: {expected_return_time(P, i):5.1f} days")

    print("\n5-step transition from Bull:")
    for s, p in zip(states, n_step(P, 5)[0]):
        print(f"  -> {s:5s}: {p:6.2%}")

    path = simulate(P, start=0, steps=20, random_state=42)
    print("\nSimulated 20-day regime path:")
    print("  " + " ".join(states[i] for i in path))
