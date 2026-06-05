<p class="lq-badges"><span class="lq-badge lq-intermediate">Intermediate</span><span class="lq-badge lq-cat">Options, Derivatives & Finance</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Discounted Cash Flow (DCF)"
    python "dcf_calculator.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Discounted%20Cash%20Flow%20%28DCF%29)

---
# Discounted Cash Flow (DCF) Calculator

This tool calculates the present value of a series of future cash flows—the basic principle behind valuing businesses, real estate, projects, and stocks!

## What is DCF?
- DCF stands for Discounted Cash Flow.
- It's a method of valuing an investment by summing its projected future cash flows, each reduced (discounted) for the time value of money.
- **Why discount?** A dollar today is worth more than a dollar tomorrow!

## Why Should I Care?
- Bankers, investors, analysts, and even exam-takers use DCF every day.
- Delivers a transparent, logical view of value, especially for stocks or projects.

## How to Use
1. Enter a list of cash flows for each future period.
2. Enter a discount rate (your opportunity cost or target return, as a decimal).
3. Call the `discounted_cash_flow()` function.

### Example
```python
from dcf_calculator import discounted_cash_flow
future_cash_flows = [1000, 1200, 1500, 2000]  # Four years of income
present_value = discounted_cash_flow(future_cash_flows, 0.08)  # 8% discount rate
print('Project Value:', present_value)
```

## Where is DCF Used?
- Stock valuation, M&A, business cases, real estate, capital budgeting, and more.

*See also: Portfolio, Bond, and Option UTILS for more hands-on finance!*


---

## Continue in Options, Derivatives & Finance

<div class="grid cards" markdown>

-   :material-chart-bell-curve: __[Advanced Options Pricing](Advanced Options Pricing.md)__

    This module covers advanced mathematical techniques for pricing financial derivatives. The focus is on models beyond the standard assumptions. Rather than assuming constant volatility, we explore dynamic and local volatility models. These models are crucial for correctly valuing exotic options and managing the risks of complex derivatives portfolios.

-   :material-chart-bell-curve: __[Black-Scholes Option Pricing](Black-Scholes Option Pricing.md)__

    This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.

-   :material-chart-bell-curve: __[Bond Price and Yield](Bond Price and Yield.md)__

    This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.

-   :material-chart-bell-curve: __[CAPM](CAPM.md)__

    CAPM is the idea that won a Nobel Prize and still anchors how the industry

-   :material-chart-bell-curve: __[Dividend Tracker](Dividend Tracker.md)__

    **This utility does NOT use any external APIs.** All data is managed locally for learning and experimentation.

-   :material-chart-bell-curve: __[Finance - Beta Calculator](Finance - Beta Calculator.md)__

    **Beta** measures how much a stock or portfolio moves compared to the overall market.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
