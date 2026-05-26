# Value at Risk (VaR) Calculator

This utility lets you estimate the potential losses on a portfolio or investment using Value at Risk (VaR), one of the most important tools in financial risk management.

## What is VaR?
- VaR tells you "how much you could lose, with a certain probability, over a set time period."
- Example: "There is a 5% chance I lose more than $1000 tomorrow on this portfolio."
- Used by banks, asset managers, hedge funds—anywhere risk needs to be measured and controlled.

## How to Use
1. Get your returns data (daily, weekly, or monthly returns).
2. Call `value_at_risk()` from `var_calculator.py`, giving it your returns and desired confidence level (like 95%).

### Example
```python
from var_calculator import value_at_risk
import numpy as np
returns = np.random.normal(0.001, 0.02, 252)
print('95% VaR:', value_at_risk(returns, confidence_level=0.95))
```

## Why Does It Matter?
- Professionals use VaR to measure the risk of stocks, funds, portfolios, and entire banks!
- Helps decide capital requirements and set trading limits

*For more financial learning, see other UTILS and Documentation modules.*
