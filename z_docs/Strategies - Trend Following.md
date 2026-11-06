# Trend Following

<p class="lq-meta">Advanced <span>/</span> Strategies <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Trend%20Following">source</a></p>


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

## Running it

```bash
cd "Strategies - Trend Following"
python "trend_following.py"
```

## More in Strategies

- [Order Execution Simulator](Order Execution Simulator.md): This tool lets you simulate buy and sell orders, track a virtual portfolio, and analyze trade performance.
- [Strategies - Backtesting Engine](Strategies - Backtesting Engine.md): A backtest answers one question: *if I had traded this rule, what would have happened?* This module is a compact, honest backtester for single-asset strategies — it converts a price series and a position series into an...
- [Strategies - Market Making](Strategies - Market Making.md): Implementation of the Avellaneda-Stoikov (2008) continuous-time market making model.
- [Strategies - Mean Reversion](Strategies - Mean Reversion.md): Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it.
- [Strategies - Momentum Trading](Strategies - Momentum Trading.md): Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market.
- [Strategies - Optimal Execution](Strategies - Optimal Execution.md): A signal tells you what to own. It does not tell you how to get there.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
