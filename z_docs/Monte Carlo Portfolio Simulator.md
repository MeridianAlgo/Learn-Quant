# Monte Carlo Portfolio Simulator

<p class="lq-meta">Advanced <span>/</span> Portfolio Management <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Monte%20Carlo%20Portfolio%20Simulator">source</a></p>


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

## Running it

```bash
cd "Monte Carlo Portfolio Simulator"
python "simulator.py"
```

## More in Portfolio Management

- [Portfolio Analytics - JavaScript](Portfolio Analytics - JavaScript.md): A portfolio is not the sum of its holdings, and the whole of portfolio theory follows from that one fact.
- [Portfolio Management](Portfolio Management.md): This folder contains utilities for portfolio management, risk analysis, and investment optimization.
- [Portfolio Management - Black Litterman](Portfolio Management - Black Litterman.md): The Black-Litterman (1990) model addresses the instability of mean-variance optimization by blending market equilibrium returns with investor views using Bayesian updating.
- [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md): Markowitz optimisation needs the inverse of the covariance matrix, and the inverse of a noisy matrix is mostly noise.
- [Portfolio Management - Risk Parity](Portfolio Management - Risk Parity.md): Risk parity builds a portfolio where every asset contributes the same amount of risk to the total — not the same amount of capital.
- [Portfolio Optimizer](Portfolio Optimizer.md): This utility helps you find the best mix of assets for a portfolio, balancing risk and return using the foundation of Modern Portfolio Theory (MPT).

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
