<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Python Fundamentals</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Python Basics - Dates and Times"
    python "dates_and_times.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20Dates%20and%20Times)

---
# Python Basics — Dates and Times

Markets run on a calendar, not a clock. Interest accrues over **days**, options
expire on a **date**, and any backtest that miscounts trading days will quietly
misreport every annualised number it produces. Python's `datetime` module gives
you the raw material; finance adds two wrinkles the standard library does not
know about — **weekends and holidays are not trading days**, and **"a year" can
mean 360, 365, or actual days** depending on the instrument.

This lesson builds the small set of date tools you actually reach for in quant
code, from first principles and with no third-party dependencies.

## Functions

| Function | Description |
|---|---|
| `parse_timestamp(value)` | Parse a date/datetime string, trying common vendor formats |
| `is_trading_day(day, holidays=None)` | True when `day` is a weekday and not a holiday |
| `next_trading_day(day, holidays=None)` | First trading day strictly after `day` |
| `trading_days_between(start, end, holidays=None)` | Count trading days in `(start, end]` |
| `add_business_days(start, n, holidays=None)` | Step `n` trading days forward or backward |
| `year_fraction(start, end, basis="ACT/365")` | Year fraction under a day-count convention |

## The pieces

- **Trading days** — weekends and exchange holidays carry no price changes, so
  realised-volatility, annualisation and accrual calculations should step over
  them. Pass your own holiday list; the functions skip Saturdays and Sundays
  automatically.
- **T+N settlement** — equities settle a fixed number of *business* days after
  the trade. `add_business_days` is exactly that count.
- **Day-count conventions** — the same two dates imply a different "year
  fraction" depending on the instrument:
  - **ACT/365** — actual days over 365; common for equity-vol and many rates.
  - **ACT/360** — actual days over 360; the money-market standard.
  - **30/360** — every month treated as 30 days; classic for corporate bonds.

## Example

```python
from dates_and_times import trading_days_between, add_business_days, year_fraction

holidays = ["2024-01-01", "2024-01-15", "2024-02-19"]

# How many trading days drove a move from Jan 1 to Mar 15?
print(trading_days_between("2024-01-01", "2024-03-15", holidays))   # 52

# T+2 settlement date
print(add_business_days("2024-03-15", 2))                           # 2024-03-19

# Accrual fraction for a money-market deposit
print(year_fraction("2024-01-01", "2024-07-01", "ACT/360"))        # 0.50556
```

## Why this matters

Annualising a daily Sharpe ratio means multiplying by `sqrt(trading days per
year)` — use 252, not 365, or your numbers will be ~17% too low. Discounting a
cash flow means turning a date gap into a year fraction under the *right*
convention. Getting these primitives right is the unglamorous foundation under
every dated calculation in the rest of this repository.

## Practical notes

- These helpers use a simple weekday + holiday-set rule. For production
  calendars (early closes, regional exchanges) you would reach for a library
  like `pandas` market calendars — but the logic here is what they implement
  underneath.
- Day-count conventions are a deep rabbit hole (30E/360, ACT/ACT ISDA, …); the
  three here cover the vast majority of teaching examples.
- These dates feed directly into `Bond Price and Yield`, `Quantitative Methods -
  TVM` and `Discounted Cash Flow (DCF)`, where year fractions drive discounting.
- Pair with [`Python Basics - Numbers`](Python Basics - Numbers.md) for
  the currency-math side of the same calculations.


---

## Continue in Python Fundamentals

<div class="grid cards" markdown>

-   :material-language-python: __[Python Basics - Comprehensions](Python Basics - Comprehensions.md)__

    Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

-   :material-language-python: __[Python Basics - Control Flow](Python Basics - Control Flow.md)__

    Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

-   :material-language-python: __[Python Basics - Essential Libraries](Python Basics - Essential Libraries.md)__

    A working quant leans on a small set of libraries for almost everything. A few of

-   :material-language-python: __[Python Basics - Functions](Python Basics - Functions.md)__

    This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

-   :material-language-python: __[Python Basics - Imports and Modules](Python Basics - Imports and Modules.md)__

    Almost every Python program begins with a few import lines. An import is how you

-   :material-language-python: __[Python Basics - NumPy](Python Basics - NumPy.md)__

    Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form. All examples use realistic financial data so the connection between the NumPy API and actual quant work is immediate.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
