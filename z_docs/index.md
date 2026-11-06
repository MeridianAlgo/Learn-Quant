---
hide:
  - navigation
  - toc
---

# Learn-Quant

<p class="lq-lede">138 small, runnable lessons on quantitative finance and the
Python it is written in. Each folder covers one idea. The README explains it,
the code implements it without leaning on a library to do the hard part, and
a test checks the code does what the README says.</p>

<p class="lq-meta">150 Python files <span>/</span> 9 JavaScript files <span>/</span> MIT licence <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant">GitHub</a></p>

## Where to start

- **New to Python.** Begin with [Python Basics - Numbers](Python Basics - Numbers.md)
  and work down the Python Fundamentals tab.
- **Comfortable with Python, new to finance.** Read
  [Quantitative Methods - Statistics](Quantitative Methods - Statistics.md), then
  [Black-Scholes Option Pricing](Black-Scholes Option Pricing.md).
- **Looking for one topic.** Press <kbd>/</kbd> to search, or scan the
  [module index](modules.md).

There are also [reading orders](learning-paths.md) for a few common goals.

## Tracks

| Track | Lessons | Covers |
|---|---:|---|
| [Python Fundamentals](Python Basics - Comprehensions.md) | 10 | Numbers, strings, functions, NumPy and pandas, using market data as the running example. |
| [Data Structures](Data Structures - Arrays.md) | 4 | Arrays, lists, dicts and sets, and when each one is the right container. |
| [Algorithms](Algorithms - Backtracking.md) | 8 | Sorting, searching, graphs and dynamic programming applied to prices and orders. |
| [Advanced Python](Advanced Python - AsyncIO.md) | 6 | Async, OOP, multiprocessing and error handling for code that has to keep running. |
| [Quantitative Methods](Quantitative Methods - Bayesian Inference.md) | 27 | Statistics, time series, stochastic processes and numerical methods, built from scratch. |
| [Options, Derivatives & Finance](Advanced Options Pricing.md) | 27 | Option pricing and Greeks, bonds and yield curves, valuation and sizing. |
| [Risk & Performance](Finance - Calmar Ratio.md) | 12 | VaR, drawdowns and the ratios used to judge whether a track record is any good. |
| [Portfolio Management](Monte Carlo Portfolio Simulator.md) | 8 | Mean variance, risk parity, Black Litterman and other ways to size a book. |
| [Strategies](Order Execution Simulator.md) | 9 | Signals, backtests and execution for the standard strategy families. |
| [AI & Machine Learning](AI Development.md) | 14 | Classifiers, clustering, trees and reinforcement learning, with the leakage traps spelled out. |
| [Market Microstructure](High Frequency Trading.md) | 3 | Order books, spreads, liquidity and the cost of actually trading. |
| [Utilities & Tools](Core Utilities.md) | 10 | Data feeds, logging, FX conversion, calendars and other plumbing. |

## Recently added

- [Portfolio Management - Hierarchical Risk Parity](Portfolio Management - Hierarchical Risk Parity.md): Markowitz optimisation needs the inverse of the covariance matrix, and the inverse of a noisy matrix is mostly noise.
- [Quantitative Methods - Kernel Density Estimation](Quantitative Methods - Kernel Density Estimation.md): A histogram of returns looks different every time you move the bin edges.
- [Market Microstructure - Liquidity Measures](Market Microstructure - Liquidity Measures.md): Every strategy pays to trade, and the size of that bill decides whether a backtest survives contact with a real market.
- [Strategies - Optimal Execution](Strategies - Optimal Execution.md): A signal tells you what to own. It does not tell you how to get there.

## Running a lesson

```bash
git clone https://github.com/MeridianAlgo/Learn-Quant
cd Learn-Quant
pip install -r requirements.txt

cd "Quantitative Methods - GARCH"
python garch.py
```

The [getting started page](getting-started.md) covers virtual environments,
the JavaScript lessons and running the tests.
