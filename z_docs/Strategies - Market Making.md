# Avellaneda-Stoikov Market Making Model

<p class="lq-meta">Advanced <span>/</span> Strategies <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Market%20Making">source</a></p>


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

## Running it

```bash
cd "Strategies - Market Making"
python "market_making.py"
```

## More in Strategies

- [Order Execution Simulator](Order Execution Simulator.md): This tool lets you simulate buy and sell orders, track a virtual portfolio, and analyze trade performance.
- [Strategies - Backtesting Engine](Strategies - Backtesting Engine.md): A backtest answers one question: *if I had traded this rule, what would have happened?* This module is a compact, honest backtester for single-asset strategies — it converts a price series and a position series into an...
- [Strategies - Mean Reversion](Strategies - Mean Reversion.md): Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it.
- [Strategies - Momentum Trading](Strategies - Momentum Trading.md): Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market.
- [Strategies - Optimal Execution](Strategies - Optimal Execution.md): A signal tells you what to own. It does not tell you how to get there.
- [Strategies - Pairs Trading](Strategies - Pairs Trading.md): This module demonstrates a statistical arbitrage strategy known as Pairs Trading.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
