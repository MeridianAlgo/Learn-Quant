<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Bond Price and Yield"
    python "bond_tools.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Bond%20Price%20and%20Yield)

---
# Bond Price and Yield Calculator

This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

## What is a Bond?
- A bond is a type of loan you give to a company or government. In return, they pay you interest ("coupons") regularly, and repay the face value at maturity.
- Bonds are a huge part of financial markets, used by everyone from governments to big investors.

## Key Formulas
- **Bond Price:** Present value of all future coupon and face value payments, discounted at the yield to maturity (YTM).
- **Yield to Maturity (YTM):** The effective annual return you'd earn if you buy the bond today and hold to maturity.

## How to Use
1. Use `bond_price()` to find fair value given face, coupon, periods, and YTM.
2. Use `estimate_ytm()` to estimate the yield given price, face, coupon, and periods.

### Example
```python
from bond_tools import bond_price, estimate_ytm
print('Bond value:', bond_price(1000, 0.05, 10, 0.04))
print('Estimated YTM:', estimate_ytm(1000, 0.05, 10, 1050))
```

## Why It Matters
- Bonds are a safe and steady part of many portfolios
- Bankers, exam takers, and investors all need these calculations
- Helps you understand time value of money and how interest rates affect prices!

*Explore more in UTILS and Documentation folders!*


---

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

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
