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
