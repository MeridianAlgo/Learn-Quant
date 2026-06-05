<p class="lq-badges"><span class="lq-badge lq-beginner">Beginner</span><span class="lq-badge lq-cat">Python Fundamentals</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Python Basics - NumPy"
    python "numpy_tutorial.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Python%20Basics%20-%20NumPy)

---
# Python Basics – NumPy

## Overview

Covers the NumPy primitives that appear in virtually every quant codebase — from vectorised return calculations to portfolio variance via the quadratic form. All examples use realistic financial data so the connection between the NumPy API and actual quant work is immediate.

## Concepts Covered
- Array creation, dtypes, shapes, and 2-D OHLC matrices
- Vectorised daily and log returns with `np.diff` and `np.log`
- Descriptive statistics: mean, std, Sharpe ratio, skewness — all without loops
- Broadcasting: applying portfolio weights across a returns matrix without explicit iteration
- Covariance matrices and annualised portfolio volatility via the quadratic form (`w @ cov @ w`)
- Boolean indexing for P&L analysis: win rate, average win/loss, profit factor

## Files
- `numpy_tutorial.py`: Annotated walkthrough script; each section prints labelled output alongside the corresponding NumPy call

## How to Run
```bash
pip install numpy
python numpy_tutorial.py
```

## Sections

| Section | What it demonstrates |
|---|---|
| Arrays and dtypes | 1-D price array, 2-D OHLC matrix, shape/ndim/dtype |
| Vectorised returns | `np.diff`, `np.log`, cumulative return |
| Descriptive statistics | Mean, vol, Sharpe, min/max, skewness — 252-day sim |
| Broadcasting | Weight × returns matrix without a Python loop |
| Covariance & portfolio variance | `np.cov`, quadratic form, annualised vol |
| Boolean indexing | Daily P&L filtering, win rate, profit factor |

## Practice Ideas
- Extend the covariance section with three real tickers loaded from CSV
- Add kurtosis to the descriptive statistics section
- Compute a rolling 20-day Sharpe using `np.lib.stride_tricks`

## Next Steps
- Continue to `Python Basics - Pandas/` for time-series resampling and signal generation on top of NumPy arrays
- Apply these primitives in `Quantitative Methods - Statistics/` and `Risk Metrics/`


---

## Continue in Python Fundamentals

<div class="grid cards" markdown>

-   :material-language-python: __[Python Basics - Comprehensions](Python Basics - Comprehensions.md)__

    Comprehensions are Python's most elegant way to transform data—replacing loops with readable, performant one-liners. This module teaches **list, dict, set comprehensions**, **generator expressions**, and **functional tools** (`map`, `filter`, `reduce`, `accumulate`) used constantly in quantitative finance for data cleaning, signal generation, and portfolio calculations.

-   :material-language-python: __[Python Basics - Control Flow](Python Basics - Control Flow.md)__

    Control flow structures (`if/elif/else`, `for`, `while`, comprehensions, `break`, `continue`) are the foundation of all algorithms. This module teaches how to make decisions, iterate through data, and build the logic patterns used in trading systems, backtests, and risk management tools.

-   :material-language-python: __[Python Basics - Functions](Python Basics - Functions.md)__

    This utility teaches Python functions - the building blocks of modular, reusable code. Learn to write efficient trading algorithms and financial tools using proper function design.

-   :material-language-python: __[Python Basics - Numbers](Python Basics - Numbers.md)__

    After completing this lesson, you'll understand:

-   :material-language-python: __[Python Basics - Pandas](Python Basics - Pandas.md)__

    Covers the Pandas patterns that power real quant research pipelines — from building a synthetic OHLCV DataFrame through rolling indicators, resampling, groupby analysis, and a simple SMA-crossover backtest. Every example is grounded in price data so the link from Pandas API to practical quant work stays concrete.

-   :material-language-python: __[Python Basics - Strings](Python Basics - Strings.md)__

    This beginner-friendly utility introduces Python string fundamentals through hands-on examples. It is perfect for newcomers following the learning path in `Documentation/Programs/level1_fundamentals.py` and looking for extra practice manipulating text data.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
