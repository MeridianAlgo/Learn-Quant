import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Machine Learning - K-Means Clustering"))
from kmeans_clustering import (
    assign_clusters,
    inertia,
    kmeans,
    silhouette_score,
    update_centroids,
)


def _three_blobs(seed=42):
    rng = np.random.default_rng(seed)
    blob = lambda cx, cy: rng.normal([cx, cy], 0.3, size=(40, 2))
    return np.vstack([blob(0, 0), blob(5, 5), blob(10, 0)])


def test_assign_clusters_nearest():
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    X = np.array([[0.1, 0.1], [9.0, 9.0]])
    assert list(assign_clusters(X, centroids)) == [0, 1]


def test_update_centroids_is_mean():
    X = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
    labels = np.array([0, 0, 1])
    centroids = update_centroids(X, labels, 2)
    assert np.allclose(centroids[0], [1.0, 1.0])
    assert np.allclose(centroids[1], [10.0, 10.0])


def test_kmeans_recovers_blob_count():
    X = _three_blobs()
    labels, centroids, wcss = kmeans(X, k=3, random_state=42)
    assert len(np.unique(labels)) == 3
    assert centroids.shape == (3, 2)


def test_kmeans_balanced_blobs():
    X = _three_blobs()
    labels, _, _ = kmeans(X, k=3, random_state=42)
    counts = sorted(np.bincount(labels).tolist())
    assert counts == [40, 40, 40]


def test_kmeans_recovers_centroids():
    X = _three_blobs()
    _, centroids, _ = kmeans(X, k=3, random_state=42)
    found = {tuple(np.round(c)) for c in centroids}
    assert {(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)} == found


def test_inertia_decreases_with_k():
    X = _three_blobs()
    w1 = kmeans(X, k=1, random_state=0)[2]
    w3 = kmeans(X, k=3, random_state=0)[2]
    assert w3 < w1


def test_inertia_matches_definition():
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    labels = np.array([0, 0])
    centroids = np.array([[1.0, 0.0]])
    # Each point is distance 1 from the centroid -> 1^2 + 1^2 = 2.
    assert abs(inertia(X, labels, centroids) - 2.0) < 1e-12


def test_silhouette_high_for_separated_blobs():
    X = _three_blobs()
    labels, _, _ = kmeans(X, k=3, random_state=42)
    assert silhouette_score(X, labels) > 0.8


def test_silhouette_single_cluster_is_zero():
    X = _three_blobs()
    assert silhouette_score(X, np.zeros(len(X), dtype=int)) == 0.0


def test_kmeans_invalid_k():
    X = _three_blobs()
    try:
        kmeans(X, k=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
