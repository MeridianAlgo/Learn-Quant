"""
K-Means Clustering
------------------
Given a few hundred stocks and their return characteristics, which ones behave
alike? K-means is the simplest answer: it partitions data into ``k`` groups so
that each point sits with the cluster whose centre (centroid) is nearest. In
quant work it groups assets into peers, compresses a correlation matrix into a
handful of regimes, or finds clusters of similar trading days.

The algorithm (Lloyd's) is a two-step loop you can hold in your head:

1. **Assign** every point to its nearest centroid.
2. **Update** each centroid to the mean of the points assigned to it.

Repeat until nothing moves. The catch is initialisation — random starts can land
in bad local optima — so this module uses **k-means++**, which spreads the
initial centroids out and dramatically improves results. We also implement the
**inertia** (within-cluster sum of squares) and a vectorised **silhouette
score** so you can choose ``k`` with the elbow and silhouette methods rather
than by guessing.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Choose k initial centroids with the k-means++ scheme.

    The first centroid is random; each subsequent one is chosen with probability
    proportional to its squared distance from the nearest existing centroid, so
    the seeds start well separated.
    """
    n = X.shape[0]
    centroids = [X[rng.integers(n)]]
    for _ in range(1, k):
        dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        total = dist_sq.sum()
        if total == 0:  # all remaining points are duplicates of a centroid
            centroids.append(X[rng.integers(n)])
            continue
        probs = dist_sq / total
        centroids.append(X[rng.choice(n, p=probs)])
    return np.array(centroids)


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Label each row of *X* with the index of its nearest centroid."""
    # Distances: (n_points, k) via broadcasting.
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recompute each centroid as the mean of its assigned points.

    Empty clusters keep a re-seeded random point so ``k`` clusters always exist.
    """
    centroids = np.empty((k, X.shape[1]))
    for j in range(k):
        members = X[labels == j]
        centroids[j] = members.mean(axis=0) if len(members) else X[np.random.randint(len(X))]
    return centroids


def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Within-cluster sum of squared distances — the loss k-means minimises.

    Lower is tighter. Plotting inertia against ``k`` gives the "elbow" used to
    pick a sensible number of clusters.
    """
    return float(np.sum((X - centroids[labels]) ** 2))


def kmeans(
    X, k: int, max_iter: int = 300, tol: float = 1e-6, n_init: int = 10, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Cluster *X* into *k* groups with k-means++ and Lloyd's algorithm.

    Runs the algorithm ``n_init`` times from different seeds and keeps the
    solution with the lowest inertia, guarding against bad local optima.

    Args:
        X: Data matrix of shape ``(n_samples, n_features)``.
        k: Number of clusters.
        max_iter: Maximum Lloyd iterations per restart.
        tol: Stop a restart when centroids move less than this.
        n_init: Number of random restarts.
        random_state: Seed for reproducibility.

    Returns:
        ``(labels, centroids, inertia)`` for the best restart.
    """
    X = np.asarray(X, dtype=float)
    if k < 1 or k > len(X):
        raise ValueError("k must be between 1 and the number of samples")
    rng = np.random.default_rng(random_state)

    best_labels, best_centroids, best_inertia = None, None, np.inf
    for _ in range(n_init):
        centroids = _kmeans_plus_plus_init(X, k, rng)
        labels = assign_clusters(X, centroids)
        for _ in range(max_iter):
            new_centroids = update_centroids(X, labels, k)
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels = assign_clusters(X, centroids)
            if shift < tol:
                break
        current = inertia(X, labels, centroids)
        if current < best_inertia:
            best_labels, best_centroids, best_inertia = labels, centroids, current
    return best_labels, best_centroids, best_inertia


def silhouette_score(X, labels) -> float:
    """Mean silhouette over all points: cluster cohesion vs. separation.

    For each point, ``a`` is its mean distance to its own cluster and ``b`` the
    mean distance to the nearest *other* cluster; its silhouette is
    ``(b - a) / max(a, b)``. The average ranges from -1 (wrong clusters) through
    0 (overlapping) to +1 (tight, well-separated) — a ``k``-selection metric
    that, unlike inertia, does not simply improve with more clusters.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    scores = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        own[i] = False
        a = dist[i, own].mean() if own.any() else 0.0
        b = min(dist[i, labels == lab].mean() for lab in unique if lab != labels[i])
        scores[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(scores.mean())


if __name__ == "__main__":
    print("K-Means Clustering")
    print("=" * 40)

    # Three well-separated blobs of (return, volatility)-like points.
    rng = np.random.default_rng(42)
    blob = lambda cx, cy: rng.normal([cx, cy], 0.35, size=(40, 2))
    X = np.vstack([blob(0, 0), blob(4, 4), blob(8, 0)])

    labels, centroids, wcss = kmeans(X, k=3, random_state=42)
    counts = np.bincount(labels)
    print(f"Clustered {len(X)} points into 3 groups of sizes {counts.tolist()}")
    print("Centroids (recovered):")
    for c in np.round(centroids[np.argsort(centroids[:, 0])], 2):
        print(f"  {c}")
    print(f"\nInertia (WCSS): {wcss:.2f}")
    print(f"Silhouette    : {silhouette_score(X, labels):.3f}  (closer to 1 is better)")

    print("\nElbow scan (inertia by k):")
    for kk in range(1, 6):
        _, _, w = kmeans(X, k=kk, random_state=42)
        print(f"  k={kk}: {w:8.2f}")
