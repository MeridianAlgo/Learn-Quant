<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Portfolio Management</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Portfolio Management - Black Litterman"
    python "black_litterman.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Portfolio%20Management%20-%20Black%20Litterman)

---
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


---

## Continue in Portfolio Management

<div class="grid cards" markdown>

-   :material-briefcase-outline: __[Monte Carlo Portfolio Simulator](Monte Carlo Portfolio Simulator.md)__

    This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!

-   :material-briefcase-outline: __[Portfolio Management](Portfolio Management.md)__

    This folder contains utilities for portfolio management, risk analysis, and investment optimization.

-   :material-briefcase-outline: __[Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md)__

    Risk parity builds a portfolio where **every asset contributes the same amount of risk** to the total — not the same amount of capital. A naive 60/40 stock/bond portfolio is ~90% *equity risk* despite being only 60% equity *capital*; risk parity fixes that imbalance.

-   :material-briefcase-outline: __[Portfolio Optimizer](Portfolio Optimizer.md)__

    This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

-   :material-briefcase-outline: __[Portfolio Tracker](Portfolio Tracker.md)__

    **This utility uses the yfinance API to fetch current prices automatically.** All other calculations and data are managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
