<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">Strategies</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Strategies - Trend Following"
    python "trend_following.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Trend%20Following)

---
# Trend Following

Trend-following: ride momentum with discipline. Backbone of CTAs and managed-futures funds (AHL, Winton, Man, MLP). Profits from extended directional moves; pays for it during chop.

## Functions

| Function | Description |
|---|---|
| `donchian_channel(high, low, window)` | Rolling N-period high/low band |
| `donchian_breakout_signal(c, h, l, entry, exit)` | Turtle-style entry/exit |
| `ma_crossover_signal(prices, fast, slow)` | Fast/slow MA crossover |
| `tsmom_signal(prices, lookback)` | Time-Series Momentum (MOP 2012) |
| `atr(high, low, close, window)` | Wilder's Average True Range |
| `atr_position_size(cap, risk, atr, k)` | Vol-targeted size |
| `trend_strength(prices, window)` | Log-price OLS slope (annualized) |

## Donchian Breakout (Turtles)

```
entry_high = max(high[t-N : t-1])
exit_low   = min(low[t-M : t-1])
long when close > entry_high; flat when close < exit_low
```

Classic windows: N=20 / M=10 (system 1), N=55 / M=20 (system 2).

## ATR Position Sizing

```
contracts = (capital * risk_per_trade) / (ATR * stop_multiple)
```

Risk a fixed dollar amount per trade — bigger ATR → smaller size.

## Example

```python
from trend_following import donchian_breakout_signal, atr, atr_position_size

pos = donchian_breakout_signal(close, high, low, 20, 10)
a = atr(high, low, close, 14)
size = atr_position_size(100_000, 0.01, a[-1], stop_atr_multiple=2.0)
```

## Practical Notes

- Trend systems lose ~60% of months but win on tail months.
- Diversify across markets (equities, rates, FX, commodities) — single-market trend is fragile.
- Use volatility targeting: bigger ATR → smaller size, keeps risk constant.
- TSMOM 12-month lookback is robust across asset classes (Moskowitz et al. 2012).


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

-   :material-trending-up: __[Strategies - Pairs Trading](Strategies - Pairs Trading.md)__

    This module demonstrates a statistical arbitrage strategy known as Pairs Trading. It identifies two assets that move together and trades the convergence of their spread. When the correlation weakens temporarily, executing trades on both assets allows for capturing profits as they revert to their historical relationship. This quantitative technique relies strictly on mathematical relationships rather than fundamental valuation.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
