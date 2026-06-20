<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Python Fundamentals</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Python Basics - Numbers"
    python "numbers_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20Numbers)

---
# Python Basics — Numbers

## Learning Objectives

After completing this lesson, you'll understand:

- **When to use integers vs. floats** — and why it matters in finance
- **Why Decimal is required for money** — not optional, required
- **How to perform basic financial calculations** — percentage change, compound interest, time conversion
- **Real-world precision problems** — and how to solve them with Python

This is foundational knowledge required for all quantitative finance work.

## What You'll Learn

### 1. **Integers vs. Floats**
   - **Integers**: Whole numbers (7, -3, 1000) used for counting
   - **Floats**: Decimal numbers (152.375, 0.08) used for measurements
   - **When it matters**: Trading 7 shares is different from trading 7.5 shares
   - **The problem**: Float arithmetic has rounding errors due to binary representation

### 2. **Decimal Module for Currency**
   - Why floats fail: `0.1 + 0.2 != 0.3` in Python, due to binary representation
   - How Decimal fixes it: Exact decimal arithmetic for money
   - **Critical rule**: Always create Decimal from strings: `Decimal("2.49")` not `Decimal(2.49)`
   - Real cost: Small rounding errors compound into massive losses in large trading systems

### 3. **Math Helpers**
   - `abs(x)` — absolute value (magnitude of change, ignore direction)
   - `round(x, decimals)` — round to N decimal places (essential for displaying money)
   - `pow(base, exponent)` — raise to a power (used in compound calculations)

### 4. **Finance Formulas (Real-World)**
   - **Percent change/return**: `(end - start) / start`
   - **Compound interest**: `FV = PV × (1 + rate)^periods`
   - **Time conversion**: `(1 + annual)^(1/12) - 1` for monthly rate (not annual/12!)

## Files

- **`numbers_tutorial.py`**: Main tutorial with heavily-commented code
  - Read the source code comments WHILE running the script
  - Each section builds on the previous one
  - Includes comparison demos (float vs. Decimal, naive vs. correct calculations)

## How to Run

```bash
python numbers_tutorial.py
```

**Best practice**:
1. Open `numbers_tutorial.py` in your editor
2. Run the script in terminal/console
3. Read each source code section, then watch the console output
4. Understand WHY each calculation works before moving on

## Key Insights

### Why Float Rounding Matters
```
Executing 100 transactions with 1% fees:
- Using float: $368.59
- Using Decimal: $369.73
- Difference: $1.14 per $1000 = 0.114% error
```
Across a high-volume trading desk, errors of this size accumulate into material amounts.

### Why `(1 + annual)^(1/12)` Not `annual/12`
```
Annual rate: 8%
Wrong: 8% / 12 = 0.667% monthly
Correct: (1.08)^(1/12) - 1 = 0.6434% monthly

Over 12 months:
- Wrong gives you 7.92% (lost 0.08%)
- Correct gives you exactly 8% ✓
```

## Practice Problems

### Problem 1: Calculate a Trading Return
```python
# Stock ABC: bought at $100, sold at $108
# What's your return?

buy_price = 100.0
sell_price = 108.0
return_pct = (sell_price - buy_price) / buy_price
print(f"Return: {return_pct:.2%}")  # Should print: 8.00%
```

### Problem 2: Future Value with Compound Interest
```python
# You invest $5000 at 6% annual interest for 10 years
# How much do you have?

principal = Decimal("5000.00")
rate = Decimal("0.06")
years = 10
future_value = principal * (1 + rate) ** years
print(f"Final amount: ${future_value}")  # Should be ~$8954.24
```

### Problem 3: Monthly Compounding
```python
# Monthly savings of $500, compounded monthly at 5% annual
# How much after 1 year?

monthly_rate = Decimal("0.05") / 12
months = 12
savings = Decimal("500.00")
total = sum(savings * (1 + monthly_rate) ** (months - i) for i in range(months))
print(f"After 1 year: ${total}")
```

## Learning Path

**Prerequisites**: None (this is a foundation module)

**Next step**: [Python Basics – Strings](Python Basics - Strings.md) to learn text processing

**Then progress to**: 
- [Data Structures – Arrays](Data Structures - Arrays.md) for working with collections
- [Python Basics – Pandas](Python Basics - Pandas.md) for working with tabular financial data

## Real-World Application

This knowledge directly applies to:
- Calculating portfolio returns and performance
- Building pricing models (Black-Scholes, DCF, etc.)
- Managing trading positions and P&L calculations
- Risk management and value-at-risk calculations
- Any quantitative finance simulation or backtest

## Common Questions

**Q: Should I always use Decimal?**
A: For calculations involving money, yes. For statistical analysis or machine learning, floats are appropriate.

**Q: Why not just use round()?**
A: `round()` masks the problem temporarily but doesn't fix it. The errors still accumulate internally.

**Q: Is 6-decimal precision enough?**
A: For most finance work, yes. For ultra-precise work (e.g., pricing derivatives), increase with `getcontext().prec = 28`.

## Further Reading

- Python `decimal` module docs: https://docs.python.org/3/library/decimal.html
- "What Every Computer Scientist Should Know About Floating-Point Arithmetic" by David Goldberg
- Real-world finance precision requirements in your exchange's documentation

---

## Continue in Python Fundamentals

<div class="grid cards" markdown>

-   :material-language-python: __[Python Basics - Comprehensions](Python Basics - Comprehensions.md)__

    Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

-   :material-language-python: __[Python Basics - Control Flow](Python Basics - Control Flow.md)__

    Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

-   :material-language-python: __[Python Basics - Dates and Times](Python Basics - Dates and Times.md)__

    Markets run on a calendar, not a clock. Interest accrues over **days**, options

-   :material-language-python: __[Python Basics - Essential Libraries](Python Basics - Essential Libraries.md)__

    A working quant leans on a small set of libraries for almost everything. A few of

-   :material-language-python: __[Python Basics - Functions](Python Basics - Functions.md)__

    This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

-   :material-language-python: __[Python Basics - Imports and Modules](Python Basics - Imports and Modules.md)__

    Almost every Python program begins with a few import lines. An import is how you

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
