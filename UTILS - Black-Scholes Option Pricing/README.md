# Black-Scholes Option Pricing Utility

This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

## What is Black-Scholes?
- Black-Scholes is a mathematical model used to estimate the fair price of European call and put options.
- Options give you the right (but not the obligation) to buy or sell a stock at a set price in the future.
- Professionals use this formula to value options and manage risk every day.

## How to Use
1. Fill in the current stock price, strike price, time to expiration, risk-free rate, and volatility.
2. Call the `black_scholes()` function from `black_scholes.py`.
3. Choose either "call" or "put" depending on your option.

### Example
```python
from black_scholes import black_scholes
S = 100     # Stock price
K = 105     # Strike price
T = 1       # Years to expiry
r = 0.03    # Risk-free rate (e.g., 3%)
sigma = 0.2 # Annual volatility (20%)
price = black_scholes(S, K, T, r, sigma, 'call')
print('Call Price:', price)
```

## Why It Matters
- Used by retail, institutional, and academic practitioners globally
- Helps make informed decisions about trading, hedging, and investing
- Required for financial certification exams and many job interviews

*For other quant finance tools and learning, see more UTILS and Documentation files!*
