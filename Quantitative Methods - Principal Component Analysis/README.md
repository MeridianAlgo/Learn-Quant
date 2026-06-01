# Principal Component Analysis (PCA)

PCA finds the orthogonal directions that explain the most variance in a dataset. In finance it powers **yield-curve decomposition** (level/slope/curvature), **statistical factor extraction**, **dimensionality reduction**, and **covariance de-noising**.

## Functions

| Function | Description |
|---|---|
| `standardize(data)` | Z-score columns to mean 0, unit variance (for correlation-matrix PCA) |
| `pca(data, n_components=None)` | Eigendecomposition → components, eigenvalues, variance ratios, scores |
| `reconstruct(scores, components, mean)` | Rebuild data from a (truncated) component set |
| `cumulative_variance(ratios)` | Running cumulative variance explained (scree / elbow analysis) |

## Key Concepts

- **PCA = eigendecomposition of the covariance matrix.** Eigenvectors are the component directions (loadings); eigenvalues are the variance each explains.
- **Yield curves**: the first three PCs are reliably interpretable as *level* (parallel shift), *slope* (steepening/flattening), and *curvature* (bowing) — together they typically explain >99% of variation.
- **Variance ratio**: `eigenvalue / total variance` tells you how much each component matters; use the cumulative sum to pick how many to keep.
- **Low-rank reconstruction**: keep the top-k components and discard the rest to de-noise a covariance matrix or compress correlated returns.

## Example

```python
import numpy as np
from pca import pca, cumulative_variance

returns = np.random.default_rng(0).normal(0, 0.01, (500, 8))
result = pca(returns, n_components=3)

print(result["explained_variance_ratio"])     # variance per PC
print(cumulative_variance(result["explained_variance_ratio"]))
print(result["scores"].shape)                  # (500, 3) factor time series
```

## Practical Notes

- Use `np.linalg.eigh` (symmetric solver), not `eig` — it is faster and returns real, orthonormal eigenvectors.
- Standardise first (`standardize`) when variables have different units, so you analyse the **correlation** matrix rather than the covariance matrix.
- Eigenvector signs are arbitrary; interpret loadings by their *relative* signs and magnitudes, not absolute sign.
