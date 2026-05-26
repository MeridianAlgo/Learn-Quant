# Portfolio Optimizer (Mean-Variance)

This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

## What is Portfolio Optimization?
- It's choosing how much of each asset (e.g., stock, fund) to hold, to get the highest return for a given risk—or lowest risk for a given return.
- The mean-variance method by Markowitz is the classic blueprint. It led to a Nobel prize!

## What Does This Module Do?
- Finds the portfolio with the highest Sharpe ratio (best risk-adjusted return)
- Uses asset expected returns & covariance matrix (how assets move together)

## How to Use
1. Enter expected annual returns as a numpy array (one for each asset).
2. Enter the covariance matrix (numpy array), describing how assets co-move.
3. Optionally, add a risk-free rate.
4. Call `mean_variance_optimizer()` for weights of the optimal portfolio.

### Example
```python
from optimizer import mean_variance_optimizer
import numpy as np
means = np.array([0.08, 0.10, 0.12])
cov = np.array([[0.04, 0.01, 0.01], [0.01, 0.09, 0.02], [0.01, 0.02, 0.16]])
w = mean_variance_optimizer(means, cov, risk_free_rate=0.03)
print('Optimal Weights:', w)
```

## Why It Matters
- Used by real portfolio managers, CFA takers, bankers, and academics
- Shows how diversification lowers risk
- Great stepping stone to deeper finance and investing topics

*See other UTILS modules for more finance learning and analysis!*
