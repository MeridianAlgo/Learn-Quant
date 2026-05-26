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
