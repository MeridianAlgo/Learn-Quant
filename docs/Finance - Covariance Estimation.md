# Robust Covariance Estimation

Sample covariance is noisy and often poorly conditioned with many assets. Shrinkage estimators blend sample covariance with a structured target for more stable portfolio optimization.

## Functions

| Function | Description |
|---|---|
| `sample_covariance(returns)` | Standard MLE covariance |
| `ledoit_wolf_shrinkage(returns)` | Analytical LW shrinkage toward scaled identity |
| `constant_correlation_shrinkage(returns)` | Shrink toward equal-correlation matrix |
| `ewma_covariance(returns, lambda_)` | RiskMetrics exponentially weighted covariance |
| `condition_number(cov)` | Ratio of max/min eigenvalue (lower = better) |

## Key Concepts

- **Curse of dimensionality**: With N assets and T observations, sample covariance has N(N+1)/2 parameters. When T < N, it's singular.
- **Ledoit-Wolf**: Finds optimal alpha blending `S` toward `mu*I`. Minimizes expected Frobenius distance to true covariance.
- **Constant Correlation**: Preserves sample variances but equalizes all correlations to the cross-sectional mean. Works well for equity portfolios.
- **EWMA**: Downweights old data — lambda=0.94 (daily) makes the half-life ~11 days. Captures volatility clustering.

## Example

```python
import numpy as np
from covariance_estimation import ledoit_wolf_shrinkage, condition_number

T, N = 252, 50
returns = np.random.randn(T, N) * 0.01
S = np.cov(returns.T)

lw = ledoit_wolf_shrinkage(returns)
print(f"Sample cov condition number: {condition_number(S):.0f}")
print(f"LW shrunk condition number:  {condition_number(lw['shrunk_cov']):.0f}")
print(f"Shrinkage intensity alpha:   {lw['alpha']:.3f}")
```

## Rule of Thumb

- T > 10N: Sample covariance is fine
- T ~ 3N: Use Ledoit-Wolf or constant correlation
- T < 2N: Consider factor-based covariance models
