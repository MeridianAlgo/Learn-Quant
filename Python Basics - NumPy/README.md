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
