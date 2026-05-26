# Pairs Trading Strategy

## Overview

This module demonstrates a statistical arbitrage strategy known as Pairs Trading. It identifies two assets that move together and trades the convergence of their spread. When the correlation weakens temporarily, executing trades on both assets allows for capturing profits as they revert to their historical relationship. This quantitative technique relies strictly on mathematical relationships rather than fundamental valuation.

## Market Execution Diagram

The following illustration maps how price boundaries dictate exact entry and exit commands across both assets.

```text
Asset A Price ---------                                  ---------
                       \     Spread Diverges            /  Spread Converges
                        \____                          /
                             \                        /
                              \                      /
Asset B Price -----------------\--------------------/-----
                               ^                    ^
                         Enter Trade           Exit Trade
                         Sell Asset A         Buy Asset A
                         Buy Asset B          Sell Asset B
```

## Key Components

1.  **Synthetic Data Generation**: Creates two correlated price pathways.
2.  **Spread Calculation**: The difference between the primary asset and the product of the hedge ratio and the secondary asset.
3.  **Standardized Normalization**: Determining how statistically significant the current spread deviation is from the historical average.
4.  **Signal Generation**: Mean reversion signals based on crossing mathematical thresholds.

## Usage

Run the script to see the generated data matrices with explicit trading signals. The numerical outputs indicate exact sizing requirements per leg of the trade.

```bash
python pairs_trading.py
```
