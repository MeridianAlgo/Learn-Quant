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
