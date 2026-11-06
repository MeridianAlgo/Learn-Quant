# Portfolio Management, Hierarchical Risk Parity

<p class="lq-meta">Advanced <span>/</span> Portfolio Management <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Management%20-%20Hierarchical%20Risk%20Parity">source</a></p>


Markowitz optimisation needs the inverse of the covariance matrix, and the
inverse of a noisy matrix is mostly noise. Hierarchical Risk Parity builds a
diversified portfolio without inverting anything, by grouping similar assets
into a tree and splitting capital down that tree in inverse proportion to
risk.

The method comes from Marcos Lopez de Prado's 2016 paper "Building
Diversified Portfolios that Outperform Out of Sample". Every step here is
written out by hand, including the clustering, so you can see exactly what
the algorithm does at each stage.

## The pieces

* `correlation_distance(corr)` turns correlations into distances with
  `sqrt((1 - rho) / 2)`, which is a true metric.
* `single_linkage(dist)` merges the two closest clusters again and again until
  one is left, returning a SciPy style linkage matrix.
* `quasi_diagonal_order(link)` reads the leaves of the tree left to right, so
  correlated assets end up next to each other.
* `cluster_variance(cov, idx)` is the variance of an inverse variance
  portfolio built from a subset of assets.
* `recursive_bisection(cov, order)` splits capital top down through the
  ordered list.
* `hrp_weights(returns)` runs the whole pipeline on a return matrix.

## The three steps

**Tree clustering.** Each asset starts as its own cluster. At every step the
two clusters whose closest members are nearest get merged. After `n - 1`
merges you have a binary tree whose shape records which assets behave alike.

**Quasi diagonalisation.** Reading the leaves of that tree in order gives a
permutation of the assets. Reorder the covariance matrix with it and the
large covariances collect into blocks along the diagonal. Nothing is
estimated in this step. It is just a sort.

**Recursive bisection.** Start with all capital on the whole ordered list.
Cut the list in half and measure the variance of each half as if it were held
on its own with inverse variance weights. Give the left half the fraction

```
alpha = 1 - var_left / (var_left + var_right)
```

and the right half the rest. Repeat inside each half until every group is a
single asset.

## Why it holds up out of sample

A mean variance optimiser treats two highly correlated assets as a hedging
opportunity and goes long one and short the other in size. When the
correlation estimate drifts a little, those positions flip. HRP never pits
assets against each other like that. Correlated names land in the same
cluster and share a single allocation, so the estimate error inside the
cluster barely matters.

The demo drops 25 days from a 750 day sample and recomputes the weights. The
largest change is a fraction of a percent. Try the same thing with an
unconstrained minimum variance portfolio and compare.

## Example

```python
import numpy as np
from hrp import hrp_weights

rng = np.random.default_rng(0)
returns = rng.normal(0, 0.01, size=(500, 5))
weights = hrp_weights(returns)
print(weights.round(3), weights.sum())
```

## Things to watch

* The bisection here splits the ordered list at its midpoint, as in the
  original paper. Later variants split at the tree's own branches instead.
  Both are defensible and they give different answers on lopsided trees.
* Single linkage is prone to chaining, where one long thin cluster absorbs
  everything. Ward or average linkage are common swaps if you see it.
* HRP still needs a covariance estimate for the variances. A shrunk estimate
  from the covariance module tends to help.
* Weights are always long only. If you need shorts, this is the wrong tool.

## Where to go next

* For the flat, non hierarchical version see
  [`Portfolio Management - Risk Parity`](Portfolio Management - Risk Parity.md).
* For the optimiser HRP is reacting against see
  [`Portfolio Optimizer`](Portfolio Optimizer.md).
* For better covariance inputs see
  [`Finance - Covariance Estimation`](Finance - Covariance Estimation.md).
* For another way to find structure in returns see
  [`Machine Learning - K-Means Clustering`](Machine Learning - K-Means Clustering.md).

---

## Running it

```bash
cd "Portfolio Management - Hierarchical Risk Parity"
python "hrp.py"
```

## More in Portfolio Management

- [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md): This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!
- [Portfolio Analytics - JavaScript](Portfolio Analytics - JavaScript.md): A portfolio is not the sum of its holdings, and the whole of portfolio theory follows from that one fact.
- [Portfolio Management](Portfolio Management.md): This folder contains utilities for portfolio management, risk analysis, and investment optimization.
- [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md): The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending market equilibrium returns with investor views using Bayesian updating.
- [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md): Risk parity builds a portfolio where every asset contributes the same amount of risk to the total — not the same amount of capital.
- [Portfolio Optimizer](Portfolio Optimizer.md): This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
