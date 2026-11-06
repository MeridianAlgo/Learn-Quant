# Portfolio Management, Hierarchical Risk Parity

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
  [`Portfolio Management - Risk Parity`](../Portfolio%20Management%20-%20Risk%20Parity/).
* For the optimiser HRP is reacting against see
  [`Portfolio Optimizer`](../Portfolio%20Optimizer/).
* For better covariance inputs see
  [`Finance - Covariance Estimation`](../Finance%20-%20Covariance%20Estimation/).
* For another way to find structure in returns see
  [`Machine Learning - K-Means Clustering`](../Machine%20Learning%20-%20K-Means%20Clustering/).
