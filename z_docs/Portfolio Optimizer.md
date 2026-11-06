# Portfolio Optimizer (Mean-Variance)

<p class="lq-meta">Advanced <span>/</span> Portfolio Management <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Optimizer">source</a></p>


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

---

## Running it

```bash
cd "Portfolio Optimizer"
python "portfolio_tutorial.py"
```

## More in Portfolio Management

- [Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md): This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!
- [Portfolio Analytics - JavaScript](Portfolio Analytics - JavaScript.md): A portfolio is not the sum of its holdings, and the whole of portfolio theory follows from that one fact.
- [Portfolio Management](Portfolio Management.md): This folder contains utilities for portfolio management, risk analysis, and investment optimization.
- [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md): The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending market equilibrium returns with investor views using Bayesian updating.
- [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md): Markowitz optimisation needs the inverse of the covariance matrix, and the inverse of a noisy matrix is mostly noise.
- [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md): Risk parity builds a portfolio where every asset contributes the same amount of risk to the total — not the same amount of capital.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
