<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Black-Scholes Option Pricing"
    python "options_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Black-Scholes%20Option%20Pricing)

---
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


---

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Bond Price and Yield](Bond Price and Yield.md)__

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   :material-chart-bell-curve: __[CAPM](CAPM.md)__

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   :material-chart-bell-curve: __[Discounted Cash Flow (DCF)](Discounted Cash Flow (DCF).md)__

    This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

-   :material-chart-bell-curve: __[Dividend Tracker](Dividend Tracker.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   :material-chart-bell-curve: __[Finance - Beta Calculator](Finance - Beta Calculator.md)__

    **Beta** measures how much a stock or portfolio moves compared to the overall market.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
