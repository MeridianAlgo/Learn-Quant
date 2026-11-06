# Pairs Trading Strategy

<p class="lq-meta">Advanced <span>/</span> Strategies <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Pairs%20Trading">source</a></p>


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

---

## Running it

```bash
cd "Strategies - Pairs Trading"
python "pairs_trading.py"
```

## More in Strategies

- [Order Execution Simulator](Order Execution Simulator.md): This tool lets you simulate buy and sell orders, track a virtual portfolio, and analyze trade performance.
- [Strategies - Backtesting Engine](Strategies - Backtesting Engine.md): A backtest answers one question: *if I had traded this rule, what would have happened?* This module is a compact, honest backtester for single-asset strategies — it converts a price series and a position series into an...
- [Strategies - Market Making](Strategies - Market Making.md): Implementation of the Avellaneda-Stoikov (2008) continuous-time market making model.
- [Strategies - Mean Reversion](Strategies - Mean Reversion.md): Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it.
- [Strategies - Momentum Trading](Strategies - Momentum Trading.md): Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market.
- [Strategies - Optimal Execution](Strategies - Optimal Execution.md): A signal tells you what to own. It does not tell you how to get there.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
