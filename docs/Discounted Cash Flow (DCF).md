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
