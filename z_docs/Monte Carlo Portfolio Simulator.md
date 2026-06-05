<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Portfolio Management</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Monte Carlo Portfolio Simulator"
    python "simulator.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Monte%20Carlo%20Portfolio%20Simulator)

---
# Monte Carlo Portfolio Simulator

This utility helps you forecast possible futures for a portfolio using random simulations—a key idea in finance, risk management, and statistics!

### What is a Monte Carlo Simulation?
- It uses repeated random sampling to simulate thousands of paths your portfolio could take.
- Helps answer: "What might my $10,000 investment be worth in 10 years, factoring in market randomness?"

## Why Is It Useful?
- See the spread of possible outcomes (best case, worst case, average)
- Plan for risk, not just averages
- Core technique used by institutional and retail investors, and in exams/interviews

## How to Use
1. Choose your `initial_investment`, an average return (`mu`), volatility (`sigma`), number of periods, and number of simulations.
2. Import and call `monte_carlo_sim()` or use the `plot_monte_carlo()` function to visualize paths.

### Example
```python
from simulator import monte_carlo_sim, plot_monte_carlo
init = 10000
mu = 0.07 / 252      # 7% yearly mean
sigma = 0.15 / (252**0.5)  # 15% yearly volatility
periods = 10 * 252
results = monte_carlo_sim(init, mu, sigma, periods)
plot_monte_carlo(init, mu, sigma, periods, simulations=100)
```

## Key Ideas
- Simulates *possible* (not guaranteed) futures—actual outcomes may differ
- Shows the importance of diversification and volatility management
- Inspires deeper learning in probability and finance

*For more info, see the docstrings in `simulator.py` and other UTILS modules!*


---

## Continue in Portfolio Management

<div class="grid cards" markdown>

-   :material-briefcase-outline: __[Portfolio Management](Portfolio Management.md)__

    This folder contains utilities for portfolio management, risk analysis, and investment optimization.

-   :material-briefcase-outline: __[Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md)__

    The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending **market equilibrium returns** with **investor views** using Bayesian updating.

-   :material-briefcase-outline: __[Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md)__

    Risk parity builds a portfolio where **every asset contributes the same amount of risk** to the total — not the same amount of capital. A naive 60/40 stock/bond portfolio is ~90% *equity risk* despite being only 60% equity *capital*; risk parity fixes that imbalance.

-   :material-briefcase-outline: __[Portfolio Optimizer](Portfolio Optimizer.md)__

    This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

-   :material-briefcase-outline: __[Portfolio Tracker](Portfolio Tracker.md)__

    **This utility uses the yfinance API to fetch current prices automatically.** All other calculations and data are managed locally for learning and experimentation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
