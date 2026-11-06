"""
Hierarchical Risk Parity
------------------------
Mean variance optimisation inverts the covariance matrix, and inverting a
noisy matrix amplifies the noise. Two assets that happen to be highly
correlated in the sample get enormous offsetting weights, and the portfolio
that looked optimal in sample falls apart the moment the estimates move.

Hierarchical Risk Parity, introduced by Marcos Lopez de Prado in 2016, never
inverts anything. It does three plain things instead.

1. Cluster the assets by correlation, so similar assets sit next to each other.
2. Reorder the covariance matrix so those clusters form blocks on the diagonal.
3. Walk down the cluster tree splitting capital between the two halves at each
   level in inverse proportion to their variance.

Every step uses only additions, divisions and sorting, so a small change in
the inputs produces a small change in the weights. That stability is the
whole point.
"""

from __future__ import annotations

import numpy as np


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Turn a correlation matrix into a proper distance matrix.

    ``d = sqrt((1 - rho) / 2)`` maps perfect correlation to 0 and perfect
    negative correlation to 1, and unlike ``1 - rho`` it satisfies the
    triangle inequality, which the clustering step relies on.
    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation matrix must be square")
    clipped = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - clipped))
    np.fill_diagonal(dist, 0.0)
    return dist


def single_linkage(dist: np.ndarray) -> np.ndarray:
    """Agglomerative clustering with single linkage, written out by hand.

    Returns a linkage matrix in the same layout SciPy uses: each row is
    ``[cluster_a, cluster_b, distance, size]``. Leaves are numbered ``0..n-1``
    and the cluster created at step ``k`` gets id ``n + k``.

    Single linkage measures the gap between two clusters as the distance
    between their closest members. It is the choice in the original paper and
    it is cheap, which is fine for portfolios of a few hundred names.
    """
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    if n < 2:
        raise ValueError("need at least two assets to cluster")
    if not np.allclose(dist, dist.T):
        raise ValueError("distance matrix must be symmetric")

    members = {i: [i] for i in range(n)}
    rows = []
    next_id = n
    while len(members) > 1:
        ids = list(members)
        best = (np.inf, None, None)
        for a_pos in range(len(ids)):
            for b_pos in range(a_pos + 1, len(ids)):
                a, b = ids[a_pos], ids[b_pos]
                gap = dist[np.ix_(members[a], members[b])].min()
                if gap < best[0]:
                    best = (gap, a, b)
        gap, a, b = best
        merged = members.pop(a) + members.pop(b)
        rows.append([float(min(a, b)), float(max(a, b)), float(gap), float(len(merged))])
        members[next_id] = merged
        next_id += 1
    return np.array(rows)


def quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Read the leaves of the cluster tree from left to right.

    Placing assets in this order puts correlated names next to each other, so
    the reordered covariance matrix has its large entries near the diagonal.
    """
    link = np.asarray(link)
    n = link.shape[0] + 1

    def leaves(node: int) -> list[int]:
        if node < n:
            return [node]
        left, right = link[node - n, :2].astype(int)
        return leaves(left) + leaves(right)

    return leaves(2 * n - 2)


def inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Weights proportional to one over each asset's variance."""
    variances = np.diag(np.asarray(cov, dtype=float))
    if np.any(variances <= 0):
        raise ValueError("every variance must be positive")
    inv = 1.0 / variances
    return inv / inv.sum()


def cluster_variance(cov: np.ndarray, idx: list[int]) -> float:
    """Variance of the inverse variance portfolio built from ``idx`` only."""
    sub = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
    w = inverse_variance_weights(sub)
    return float(w @ sub @ w)


def recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Split capital top down through the ordered assets.

    At each step the current group is cut in half. The half with the lower
    variance gets the larger share, in the ratio
    ``alpha = 1 - var_left / (var_left + var_right)``.
    """
    cov = np.asarray(cov, dtype=float)
    weights = np.ones(cov.shape[0])
    groups = [list(order)]
    while groups:
        next_groups = []
        for group in groups:
            if len(group) < 2:
                continue
            half = len(group) // 2
            left, right = group[:half], group[half:]
            var_left = cluster_variance(cov, left)
            var_right = cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_groups.extend([left, right])
        groups = next_groups
    return weights


def hrp_weights(returns: np.ndarray) -> np.ndarray:
    """Full pipeline from a ``(time, assets)`` return matrix to HRP weights."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("returns must be a 2D array with at least two assets")
    if returns.shape[0] < 3:
        raise ValueError("need at least three observations")
    cov = np.cov(returns, rowvar=False)
    corr = np.corrcoef(returns, rowvar=False)
    link = single_linkage(correlation_distance(corr))
    order = quasi_diagonal_order(link)
    return recursive_bisection(cov, order)


def _simulate_blocks(n_obs: int = 750, seed: int = 7) -> np.ndarray:
    """Two blocks of three correlated assets plus one lone low vol asset."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 0.012, n_obs)
    f2 = rng.normal(0, 0.020, n_obs)
    block_a = [f1 + rng.normal(0, 0.004, n_obs) for _ in range(3)]
    block_b = [f2 + rng.normal(0, 0.006, n_obs) for _ in range(3)]
    loner = rng.normal(0, 0.005, n_obs)
    return np.column_stack(block_a + block_b + [loner])


if __name__ == "__main__":
    rets = _simulate_blocks()
    names = ["A1", "A2", "A3", "B1", "B2", "B3", "Bond"]

    corr = np.corrcoef(rets, rowvar=False)
    link = single_linkage(correlation_distance(corr))
    order = quasi_diagonal_order(link)
    print("Cluster order:", [names[i] for i in order])

    hrp = hrp_weights(rets)
    ivp = inverse_variance_weights(np.cov(rets, rowvar=False))
    print(f"\n{'Asset':<6}{'HRP':>9}{'Inv var':>10}")
    for name, h, i in zip(names, hrp, ivp):
        print(f"{name:<6}{h:>9.3f}{i:>10.3f}")

    # Show how little the weights move when the sample shifts slightly.
    shifted = hrp_weights(rets[25:])
    print(f"\nLargest weight change after dropping 25 days: {np.abs(shifted - hrp).max():.4f}")
