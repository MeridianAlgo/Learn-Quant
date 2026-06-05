<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Strategies</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Order Execution Simulator"
    python "order_execution_simulator.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Order%20Execution%20Simulator)

---
# Order Execution Simulator Utility (NO API)

**This utility does NOT use any external APIs.** All trades and portfolio data are managed locally for learning and experimentation.

This tool lets you simulate buy and sell orders, track a virtual portfolio, and analyze trade performance. All prices are entered manually.

## Features
- Simulate buy and sell orders for any asset (stocks, crypto, etc.)
- Track cash balance, holdings, and trade history
- Calculate realized and unrealized P&L
- Support for market and limit orders (simulated)
- Save and load your virtual portfolio and trade history
- CLI interface (Python script)
- **Beginner-friendly:** All code is commented for learning

## Requirements
- Python 3.7+
- No external libraries required (uses only Python standard library)

## Setup
1. Copy `order_execution_simulator.py` to your desired folder.
2. Open a terminal in that folder.

## Usage Workflow (Step-by-Step)
1. Run the script:
   ```sh
   python order_execution_simulator.py
   ```
2. Follow the menu prompts:
   - Place a buy or sell order (market or limit)
   - View portfolio and cash balance
   - View trade history and P&L
   - Save/load portfolio and trades
   - Exit when done.

**No real market data is used. This is for learning only!**

## Example Session
```
Welcome to the Order Execution Simulator!
1. Place order
2. View portfolio
3. View trade history
4. Save
5. Load
6. Exit
Enter your choice: 1
Order type (buy/sell): buy
Asset: TSLA
Quantity: 5
Price: 700
Order executed! Portfolio updated.
```

## Learning Notes
- **No API:** All calculations and data are managed in Python, so you can see and modify the logic yourself.
- **How does it work?** The code is structured with classes and functions, with comments explaining each step.
- **How can you extend it?** Try adding support for commissions, or tracking portfolio value over time!

## License
MIT


---

## Continue in Strategies

<div class="grid cards" markdown>

-   :material-trending-up: __[Strategies - Backtesting Engine](Strategies - Backtesting Engine.md)__

    A backtest answers one question: *if I had traded this rule, what would have

-   :material-trending-up: __[Strategies - Market Making](Strategies - Market Making.md)__

    Implementation of the **Avellaneda-Stoikov (2008)** continuous-time market making model. A dealer posts bid/ask quotes to maximize expected PnL while penalizing inventory accumulation.

-   :material-trending-up: __[Strategies - Mean Reversion](Strategies - Mean Reversion.md)__

    Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it. While Momentum strategies bet on *continuation*, Mean Reversion strategies bet on *reversal* — buying when something is "too cheap" and selling when it is "too expensive" relative to recent history.

-   :material-trending-up: __[Strategies - Momentum Trading](Strategies - Momentum Trading.md)__

    Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market. The core philosophy is "buy high, sell higher." If an asset's price is rising strongly, momentum traders assume it will continue to rise.

-   :material-trending-up: __[Strategies - Pairs Trading](Strategies - Pairs Trading.md)__

    This module demonstrates a statistical arbitrage strategy known as Pairs Trading. It identifies two assets that move together and trades the convergence of their spread. When the correlation weakens temporarily, executing trades on both assets allows for capturing profits as they revert to their historical relationship. This quantitative technique relies strictly on mathematical relationships rather than fundamental valuation.

-   :material-trending-up: __[Strategies - Statistical Arbitrage](Strategies - Statistical Arbitrage.md)__

    This module demonstrates a basic Statistical Arbitrage strategy, specifically pairs trading.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
