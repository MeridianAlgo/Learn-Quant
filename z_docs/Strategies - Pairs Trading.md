<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Strategies</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Strategies - Pairs Trading"
    python "pairs_trading.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Pairs%20Trading)

---
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


---

## Continue in Strategies

<div class="grid cards" markdown>

-   :material-trending-up: __[Order Execution Simulator](Order Execution Simulator.md)__

    **This utility does NOT use any external APIs.** All trades and portfolio data are managed locally for learning and experimentation.

-   :material-trending-up: __[Strategies - Backtesting Engine](Strategies - Backtesting Engine.md)__

    A backtest answers one question: *if I had traded this rule, what would have

-   :material-trending-up: __[Strategies - Market Making](Strategies - Market Making.md)__

    Implementation of the **Avellaneda-Stoikov (2008)** continuous-time market making model. A dealer posts bid/ask quotes to maximize expected PnL while penalizing inventory accumulation.

-   :material-trending-up: __[Strategies - Mean Reversion](Strategies - Mean Reversion.md)__

    Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it. While Momentum strategies bet on *continuation*, Mean Reversion strategies bet on *reversal* — buying when something is "too cheap" and selling when it is "too expensive" relative to recent history.

-   :material-trending-up: __[Strategies - Momentum Trading](Strategies - Momentum Trading.md)__

    Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market. The core philosophy is "buy high, sell higher." If an asset's price is rising strongly, momentum traders assume it will continue to rise.

-   :material-trending-up: __[Strategies - Statistical Arbitrage](Strategies - Statistical Arbitrage.md)__

    This module demonstrates a basic Statistical Arbitrage strategy, specifically pairs trading.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
