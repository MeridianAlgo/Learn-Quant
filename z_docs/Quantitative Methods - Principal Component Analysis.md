<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Quantitative Methods</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Quantitative Methods - Principal Component Analysis"
    python "pca.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Quantitative%20Methods%20-%20Principal%20Component%20Analysis)

---
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


---

## Continue in Quantitative Methods

<div class="grid cards" markdown>

-   :material-function-variant: __[Quantitative Methods - Bayesian Inference](Quantitative Methods - Bayesian Inference.md)__

    A strategy wins 7 of its first 10 trades. Is its true win rate 70%? Almost

-   :material-function-variant: __[Quantitative Methods - Bootstrap](Quantitative Methods - Bootstrap.md)__

    The bootstrap estimates the sampling distribution of **any** statistic by resampling the observed data with replacement — no normality assumption required. It is the honest way to put confidence intervals around backtest metrics like Sharpe ratio, mean return, or maximum drawdown.

-   :material-function-variant: __[Quantitative Methods - Cointegration](Quantitative Methods - Cointegration.md)__

    Cointegration: two non-stationary series whose **linear combination is stationary**. Backbone of statistical arbitrage and pairs trading.

-   :material-function-variant: __[Quantitative Methods - Copulas](Quantitative Methods - Copulas.md)__

    This module demonstrates the concept of Copulas, specifically the Gaussian Copula, used in quantitative finance to model the dependency structure between multivariate random variables.

-   :material-function-variant: __[Quantitative Methods - Extreme Value Theory](Quantitative Methods - Extreme Value Theory.md)__

    Most risk models assume returns are normally distributed. They are not —

-   :material-function-variant: __[Quantitative Methods - Factor Models](Quantitative Methods - Factor Models.md)__

    Factor models explain asset returns as a linear combination of systematic **factors** plus a stock-specific residual. The **Fama-French 3-Factor Model (1992)** extended CAPM by adding two well-documented risk premia: the **Size premium** (SMB) and the **Value premium** (HML), dramatically improving the explanation of cross-sectional stock returns.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
