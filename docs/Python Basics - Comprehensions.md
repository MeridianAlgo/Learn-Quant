# Python Basics – Comprehensions

## Overview

Covers Python's concise data-transformation tools — list, dict, and set comprehensions, generator expressions, `map`, `filter`, `functools.reduce`, and `itertools.accumulate` — all applied to quantitative finance workflows. These patterns appear constantly in professional quant code and replace verbose loops with readable, performant one-liners.

## Concepts Covered
- List comprehensions: ticker normalisation, price filtering, daily return calculation
- Dict and set comprehensions: normalising symbol dicts, deduplicating trade logs, building sector-to-stock maps
- Generator expressions: lazy return computation over large price series, memory-efficient aggregation
- `map` and `filter`: cumulative returns, strong-signal extraction, ticker formatting
- `functools.reduce`: compounding a sequence of daily returns to a total return
- `itertools.accumulate`: building an equity curve from a starting portfolio value
- Nested comprehensions: computing a correlation matrix without NumPy

## Files
- `comprehensions_tutorial.py`: Annotated walkthrough script; each section prints labelled output showing the before/after of each transformation

## How to Run
```bash
python comprehensions_tutorial.py
```
No external dependencies — uses only the Python standard library.

## Sections

| Section | What it demonstrates |
|---|---|
| List comprehensions | Ticker cleaning, price filter, per-day return list |
| Dict and set comprehensions | Normalised price dict, unique symbols, sector map |
| Generator expressions | Lazy daily returns, sum of positive returns |
| Map and filter | Cumulative returns, signal filtering, formatted tickers |
| Reduce and accumulate | Compounded return, $10,000 equity curve |
| Nested comprehensions | Pearson correlation matrix for 3 assets |

## Practice Ideas
- Rewrite the correlation matrix section using a generator expression inside `accumulate`
- Add a comprehension that zips two price series and computes spread
- Replace the `reduce` compounding with `math.prod` and benchmark both

## Next Steps
- These patterns combine naturally with NumPy in `Python Basics - NumPy/`
- See them applied at scale in `Data Processing/` and `Strategies - Statistical Arbitrage/`
