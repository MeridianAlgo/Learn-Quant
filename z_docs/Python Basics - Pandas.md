<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Python Fundamentals</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Python Basics - Pandas"
    python "pandas_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20Pandas)

---
# Python Basics – Pandas

## Overview

Covers the Pandas patterns that power real quant research pipelines — from building a synthetic OHLCV DataFrame through rolling indicators, resampling, groupby analysis, and a simple SMA-crossover backtest. Every example is grounded in price data so the link from Pandas API to practical quant work stays concrete.

## Concepts Covered
- Building a date-indexed OHLCV DataFrame with `pd.bdate_range`
- `pct_change`, `shift`, and log returns on a Series
- Rolling windows: SMA-20, Bollinger Bands, rolling Sharpe ratio
- Resampling daily OHLCV to weekly bars with `resample().agg()`
- `groupby` by day-of-week to surface return seasonality patterns
- Signal generation: SMA crossover with `np.where`, vectorised P&L, strategy vs buy-and-hold comparison

## Files
- `pandas_tutorial.py`: Annotated walkthrough script; each section prints labelled DataFrame output alongside the corresponding Pandas call

## How to Run
```bash
pip install pandas numpy
python pandas_tutorial.py
```

## Sections

| Section | What it demonstrates |
|---|---|
| DataFrame basics | Shape, head, dtypes, describe on a synthetic 60-day OHLCV frame |
| Returns and rolling windows | `pct_change`, `rolling().mean/std`, Bollinger Bands, rolling Sharpe |
| Resampling | Daily → weekly OHLC aggregation with named agg columns |
| Groupby by weekday | Day-of-week return stats as a bias check |
| Signal generation | SMA-5/20 crossover, shifted signal, strategy vs buy-and-hold |

## Practice Ideas
- Load real price data via `pandas_datareader` or a CSV and replace the synthetic DataFrame
- Add a drawdown column using `cummax()` and `cummin()`
- Extend the groupby section to group by calendar month

## Next Steps
- Apply these patterns to real data in `Market Data/`
- Move into more advanced modelling in `Quantitative Methods - Time Series/`
- See `Strategies - Momentum Trading/` for production-grade signal generation


---

## Continue in Python Fundamentals

<div class="grid cards" markdown>

-   :material-language-python: __[Python Basics - Comprehensions](Python Basics - Comprehensions.md)__

    Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

-   :material-language-python: __[Python Basics - Control Flow](Python Basics - Control Flow.md)__

    Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

-   :material-language-python: __[Python Basics - Functions](Python Basics - Functions.md)__

    This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

-   :material-language-python: __[Python Basics - NumPy](Python Basics - NumPy.md)__

    Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form. All examples use realistic financial data so the connection between the NumPy API and actual quant work is immediate.

-   :material-language-python: __[Python Basics - Numbers](Python Basics - Numbers.md)__

    After completing this lesson, you'll understand:

-   :material-language-python: __[Python Basics - Strings](Python Basics - Strings.md)__

    This beginner-friendly utility introduces Python string fundamentals through hands-on examples. It is perfect for newcomers following the learning path in `Documentation/Programs/level1_fundamentals.py` and looking for extra practice manipulating text data.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
