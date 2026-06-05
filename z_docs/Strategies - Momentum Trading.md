<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Strategies</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Strategies - Momentum Trading"
    python "momentum_strategy.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Momentum%20Trading)

---
# Strategies – Momentum Trading

## Overview

Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market. The core philosophy is "buy high, sell higher." If an asset's price is rising strongly, momentum traders assume it will continue to rise.

This utility implements a basic **Trend-Following Momentum Strategy** using Rate of Change (ROC) and Moving Averages.

## Key Concepts

### **Momentum**
- Measures the speed or velocity of price changes.
- **Rate of Change (ROC)**: The percentage change between current price and price $n$ periods ago.
- Positive Momentum suggests an Uptrend; Negative Momentum suggests a Downtrend.

### **Trend Filtering**
- Momentum signals can be false in choppy markets.
- **Moving Averages (SMA/EMA)** are often used as a filter.
- Rule: Only take Long positions if `Price > SMA` (price is above the long-term trend).

## Logic Implemented

We combine two signals:
1. **Momentum**: ROC(20) > 0
2. **Trend**: Price > SMA(50)

**Signal Logic:**
- **Enter Long (1)**: When Momentum is positive AND Price is above the 50-period SMA.
- **Exit/Neutral (0)**: When Momentum turns negative OR Price drops below the SMA.

## Files
- `momentum_strategy.py`: Logic for generating synthetic data, calculating indicators, generating signals, and running a simple backtest.

## How to Run
```bash
python momentum_strategy.py
```

## Financial Applications

### 1. Cross-Sectional Momentum
- Buying the top N performing stocks in an index and shorting the bottom N.
- Using Relative Strength (not RSI) to compare assets.

### 2. Time-Series Momentum
- Focusing on a single asset's own history (Trend Following).
- Managed Futures / CTAs implementation.

### 3. Risk Management
- Momentum strategies often suffer from "Momentum Crashes" (sudden reversals).
- Volatility scaling is often used to manage position size.

## Best Practices

- **Lookback Period**: The choice of '$n$' (e.g., 12 months, 6 months, 20 days) drastically affects performance. Longer periods reduce noise but lag trend turning points.
- **Transaction Costs**: Frequent trading in momentum strategies can erode profits.
- **Diversification**: Apply momentum across multiple unconnected assets (Equities, Commodities, FX) to smooth returns.

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

-   :material-trending-up: __[Strategies - Pairs Trading](Strategies - Pairs Trading.md)__

    This module demonstrates a statistical arbitrage strategy known as Pairs Trading. It identifies two assets that move together and trades the convergence of their spread. When the correlation weakens temporarily, executing trades on both assets allows for capturing profits as they revert to their historical relationship. This quantitative technique relies strictly on mathematical relationships rather than fundamental valuation.

-   :material-trending-up: __[Strategies - Statistical Arbitrage](Strategies - Statistical Arbitrage.md)__

    This module demonstrates a basic Statistical Arbitrage strategy, specifically pairs trading.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
