# Black-Scholes Option Pricing Utility

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Black-Scholes%20Option%20Pricing">source</a></p>


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

---

## Running it

```bash
cd "Black-Scholes Option Pricing"
python "options_tutorial.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md): This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.
- [Finance - Beta Calculator](Finance - Beta Calculator.md): Beta measures how much a stock or portfolio moves compared to the overall market.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
