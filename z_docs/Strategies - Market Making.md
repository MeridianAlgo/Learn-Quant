<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Strategies</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Strategies - Market Making"
    python "market_making.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Market%20Making)

---
# Avellaneda-Stoikov Market Making Model

Implementation of the **Avellaneda-Stoikov (2008)** continuous-time market making model. A dealer posts bid/ask quotes to maximize expected PnL while penalizing inventory accumulation.

## Functions

| Function | Description |
|---|---|
| `reservation_price(mid, q, T, t, sigma, gamma)` | Inventory-adjusted mid price |
| `optimal_spread(T, t, sigma, gamma, kappa)` | Optimal total bid-ask spread |
| `bid_ask_quotes(...)` | Both quotes + spread in one call |
| `simulate_market_maker(...)` | Full simulation with PnL and inventory tracking |

## Key Concepts

- **Reservation price**: `r = S - q * gamma * sigma² * (T - t)`. Long inventory → quote lower to attract sellers.
- **Optimal spread**: Balances adverse selection risk (sigma) vs. order arrival intensity (kappa).
- **Inventory risk**: The dealer must manage directional exposure from unbalanced fills.
- **gamma**: Risk aversion parameter. High gamma → wider spreads, more aggressive inventory management.

## Parameters

| Param | Typical value | Meaning |
|---|---|---|
| `sigma` | 1–5 | Asset volatility per unit time |
| `gamma` | 0.01–1.0 | Risk aversion (higher = more conservative) |
| `kappa` | 1–5 | Order arrival intensity (higher = more liquid) |

## Example

```python
from market_making import bid_ask_quotes, simulate_market_maker

quotes = bid_ask_quotes(mid_price=100, inventory=5, T=1.0, t=0.5, sigma=2.0, gamma=0.1, kappa=1.5)
# {'bid': 99.3, 'ask': 101.0, 'reservation_price': 100.15, 'spread': 1.7}

result = simulate_market_maker(S0=100, sigma=2.0, gamma=0.1, kappa=1.5, seed=42)
```


---

## Continue in Strategies

<div class="grid cards" markdown>

-   :material-trending-up: __[Order Execution Simulator](Order Execution Simulator.md)__

    **This utility does NOT use any external APIs.** All trades and portfolio data are managed locally for learning and experimentation.

-   :material-trending-up: __[Strategies - Backtesting Engine](Strategies - Backtesting Engine.md)__

    A backtest answers one question: *if I had traded this rule, what would have

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
