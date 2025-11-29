# Sharpe and Sortino Ratio Calculator

This utility offers easy-to-use Python functions to calculate Sharpe and Sortino ratios for financial returns. These ratios help you understand whether a series of investment returns is attractive on a risk-adjusted basis.

## What are the Sharpe and Sortino Ratios?
**Sharpe Ratio** measures how much excess return you get for each unit of total risk you take (as measured by volatility).
**Sortino Ratio** is similar, but only counts downside risk, ignoring upside swings.

- **Higher values** mean better risk-adjusted returns.
- Used by professional and retail investors to evaluate stock, fund, and portfolio performance.

## How to Use
1. Make sure you have Python and `numpy` installed.
2. Put your list or array of returns (like daily or monthly returns) in the code.
3. Call the `sharpe_ratio()` or `sortino_ratio()` function from the `ratio_calculator.py` file.
4. Optionally, set the risk-free rate and number of periods per year.

### Example
```python
from ratio_calculator import sharpe_ratio, sortino_ratio
import numpy as np

daily_returns = np.random.normal(0.001, 0.01, 252)
print("Sharpe Ratio:", sharpe_ratio(daily_returns))
print("Sortino Ratio:", sortino_ratio(daily_returns))
```

## Why Does This Matter?
- Compare the quality of investments, not just raw returns.
- Identify if an asset rewards you for the risk you take.
- Learn core principles of risk management and portfolio analysis.

*For more finance learning, check /Documentation or see other UTILS modules!*
