# Pairs Trading Strategy

## Overview
This module demonstrates a statistical arbitrage strategy known as Pairs Trading. It identifies two assets that move together (cointegrated) and trades the convergence of their spread.

## Key Components
1.  **Synthetic Data Generation**: Creates two correlated random walk series.
2.  **Spread Calculation**: `Spread = Y - (HedgeRatio * X)`
3.  **Z-Score Normalization**: determining how statistically significant the current spread deviation is.
4.  **Signal Generation**: Mean reversion signals based on Z-Score thresholds.

## Usage
Run the script to see the generated dataframe with signals.

```bash
python pairs_trading.py
```
