# Discounted Cash Flow (DCF) Calculator

<p class="lq-meta">Intermediate <span>/</span> Options, Derivatives & Finance <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Discounted%20Cash%20Flow%20%28DCF%29">source</a></p>


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

## Running it

```bash
cd "Discounted Cash Flow (DCF)"
python "dcf_calculator.py"
```

## More in Options, Derivatives & Finance

- [Advanced Options Pricing](Advanced Options Pricing.md): This module covers advanced mathematical techniques for pricing financial derivatives.
- [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md): This module lets you price basic stock options (calls and puts) using the Black-Scholes formula, a foundation of modern financial analysis.
- [Bond Price and Yield](Bond Price and Yield.md): This utility lets you calculate the fair price of a bond or estimate its yield to maturity (YTM), two of the most basic (and important!) ideas in investing.
- [CAPM](CAPM.md): CAPM is the idea that won a Nobel Prize and still anchors how the industry thinks about return and risk.
- [Dividend Tracker](Dividend Tracker.md): This tool lets you track upcoming dividends for a list of stocks, calculate expected income, and view a dividend calendar.
- [Finance - Beta Calculator](Finance - Beta Calculator.md): Beta measures how much a stock or portfolio moves compared to the overall market.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
