# Strategies - Statistical Arbitrage

This module demonstrates a basic Statistical Arbitrage strategy, specifically pairs trading.

## Concepts
- **Cointegration:** A statistical property of a collection of time series variables. If two or more series are themselves non-stationary, but a linear combination of them is stationary, then the series are said to be cointegrated.
- **Pairs Trading:** Identifying two cointegrated assets and trading the spread between them. When the spread widens beyond a historical norm (measured by z-score), you short the outperforming asset and long the underperforming one, betting the spread will revert to the mean.

## Example
Run `python stat_arb.py` to see a demonstration of testing cointegration and generating entry/exit signals based on the z-score of the spread.
