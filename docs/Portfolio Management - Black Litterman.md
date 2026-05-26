# Black-Litterman Portfolio Optimization

The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending **market equilibrium returns** with **investor views** using Bayesian updating.

## Functions

| Function | Description |
|---|---|
| `market_implied_returns(cov, weights, lambda)` | Reverse optimize: implied returns from market portfolio |
| `black_litterman(cov, weights, P, Q, omega, tau, lambda)` | Bayesian blend of equilibrium + views |
| `bl_optimal_weights(bl_returns, cov, lambda)` | Mean-variance weights from BL posterior |

## Key Concepts

- **Problem with MVO**: Small changes in expected return inputs produce wildly different (often extreme) optimal portfolios.
- **Equilibrium returns (Pi)**: `Pi = lambda * Sigma * w_mkt` — back out what the market is "pricing in."
- **Views matrix P**: Each row encodes one view. `[1, -1, 0, 0]` = "asset 1 outperforms asset 2."
- **tau**: Scales uncertainty of equilibrium priors. Typically 0.01–0.05.
- **Omega**: Diagonal matrix of view uncertainty. Larger = less confident in that view.

## Example

```python
import numpy as np
from black_litterman import black_litterman, bl_optimal_weights

# View: US equity outperforms international by 2%
P = np.array([[1, -1, 0, 0]])
Q = np.array([0.02])

result = black_litterman(cov, market_weights, P, Q)
weights = bl_optimal_weights(result["posterior_returns"], result["posterior_covariance"])
```

## Why It Works

By starting from market-cap weights (which are efficient by definition if markets are efficient), BL produces sensible portfolios even with few views. Views only tilt allocations where the manager has genuine insight.
