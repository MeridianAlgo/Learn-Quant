"""
Principal Component Analysis (PCA) for Finance
-----------------------------------------------
PCA finds the orthogonal directions ("principal components") that explain the
most variance in a dataset. In quantitative finance it is the workhorse for:

- Yield-curve decomposition: the first three PCs of bond yields are famously
  interpretable as *level*, *slope*, and *curvature*.
- Risk factor extraction: reducing a large correlated universe of stock returns
  to a handful of statistical factors.
- Dimensionality reduction before regression or ML to avoid multicollinearity.
- De-noising covariance matrices (keep the top PCs, discard the noisy tail).

The mathematics: PCA is the eigendecomposition of the covariance (or
correlation) matrix. Eigenvectors are the component directions; eigenvalues are
the variance explained by each. This module implements PCA from first
principles with NumPy so the linear algebra is fully visible — no black boxes.
"""

from typing import Optional

import numpy as np


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Z-score each column to mean 0 and unit variance.

    Standardising before PCA puts every variable on the same scale, so a
    high-variance series does not dominate purely because of its units. Use
    this when running PCA on the *correlation* matrix.

    Args:
        data: TxN array (T observations, N variables).

    Returns:
        np.ndarray: Standardised TxN array.
    """
    data = np.asarray(data, dtype=float)
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero on constant columns
    return (data - mean) / std


def pca(data: np.ndarray, n_components: Optional[int] = None):
    """
    Run PCA via eigendecomposition of the covariance matrix.

    Args:
        data: TxN array (T observations, N variables). Should be de-meaned
            or standardised by the caller depending on the desired matrix.
        n_components: Number of components to keep (default: all N).

    Returns:
        dict with keys:
            - "components": Nxk matrix; column j is the j-th eigenvector (loading).
            - "explained_variance": length-k eigenvalues (variance per component).
            - "explained_variance_ratio": fraction of total variance per component.
            - "scores": Txk projection of the data onto the components.
    """
    data = np.asarray(data, dtype=float)
    # De-mean so the covariance is centred (PCA is about variance, not level).
    centered = data - data.mean(axis=0)
    cov = np.cov(centered, rowvar=False)

    # eigh is for symmetric matrices: returns ascending eigenvalues + orthonormal
    # eigenvectors. We reverse to get them in descending order of variance.
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    total_var = np.sum(np.diag(cov))
    scores = centered @ eigvecs  # project observations onto component axes

    return {
        "components": eigvecs,
        "explained_variance": eigvals,
        "explained_variance_ratio": eigvals / total_var,
        "scores": scores,
    }


def reconstruct(scores: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Rebuild the original data from a (possibly truncated) set of components.

    Keeping only the top components gives a low-rank, de-noised approximation of
    the original series — the basis of PCA-based covariance cleaning.

    Args:
        scores: Txk projected scores.
        components: Nxk loading matrix used to produce the scores.
        mean: length-N column means that were subtracted before projection.

    Returns:
        np.ndarray: TxN reconstructed data.
    """
    return scores @ components.T + mean


def cumulative_variance(explained_variance_ratio: np.ndarray) -> np.ndarray:
    """
    Cumulative fraction of variance explained as components are added.

    Useful for the "elbow" / scree decision of how many components to keep.

    Args:
        explained_variance_ratio: per-component variance fractions.

    Returns:
        np.ndarray: running cumulative sum.
    """
    return np.cumsum(explained_variance_ratio)


if __name__ == "__main__":
    print("Principal Component Analysis — Yield Curve Demo")
    print("=" * 48)

    rng = np.random.default_rng(42)
    # Simulate daily changes in yields across 6 maturities driven by three
    # latent factors: parallel shifts (level), steepening (slope), and bowing
    # (curvature). This mirrors how real yield curves behave.
    n_days = 750
    maturities = np.array([0.25, 1, 2, 5, 10, 30])
    level = rng.normal(0, 0.06, n_days)[:, None] * np.ones_like(maturities)
    slope = rng.normal(0, 0.03, n_days)[:, None] * (maturities - maturities.mean())
    curve = rng.normal(0, 0.02, n_days)[:, None] * (maturities - maturities.mean()) ** 2
    noise = rng.normal(0, 0.005, (n_days, len(maturities)))
    yield_changes = level + slope * 0.01 + curve * 0.001 + noise

    result = pca(yield_changes)
    evr = result["explained_variance_ratio"]
    cum = cumulative_variance(evr)

    print("\nVariance explained by each principal component:")
    names = ["PC1 (level)", "PC2 (slope)", "PC3 (curvature)"]
    for i in range(3):
        print(f"  {names[i]:18s}: {evr[i]:6.2%}   cumulative {cum[i]:6.2%}")
    print(f"\nFirst 3 PCs capture {cum[2]:.2%} of all yield-curve variation.")

    print("\nPC1 loadings (should all share the same sign — a level shift):")
    pc1 = result["components"][:, 0]
    for m, load in zip(maturities, pc1):
        print(f"  {m:5.2f}y: {load:+.3f}")
