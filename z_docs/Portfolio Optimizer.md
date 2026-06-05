<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Portfolio Management</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Portfolio Optimizer"
    python "portfolio_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Optimizer)

---
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


---

## Continue in Portfolio Management

<div class="grid cards" markdown>

-   :material-briefcase-outline: __[Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md)__

    This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!

-   :material-briefcase-outline: __[Portfolio Management](Portfolio Management.md)__

    This folder contains utilities for portfolio management, risk analysis, and investment optimization.

-   :material-briefcase-outline: __[Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md)__

    The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending **market equilibrium returns** with **investor views** using Bayesian updating.

-   :material-briefcase-outline: __[Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md)__

    Risk parity builds a portfolio where **every asset contributes the same amount of risk** to the total — not the same amount of capital. A naive 60/40 stock/bond portfolio is ~90% *equity risk* despite being only 60% equity *capital*; risk parity fixes that imbalance.

-   :material-briefcase-outline: __[Portfolio Tracker](Portfolio Tracker.md)__

    **This utility uses the yfinance API to fetch current prices automatically.** All other calculations and data are managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
