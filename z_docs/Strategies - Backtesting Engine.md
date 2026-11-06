# Vectorised Backtesting Engine

<p class="lq-meta">Advanced <span>/</span> Strategies <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Backtesting%20Engine">source</a></p>


A backtest answers one question: *if I had traded this rule, what would have
happened?* This module is a compact, honest backtester for single-asset
strategies — it converts a price series and a position series into an equity
curve and the performance statistics every strategy report quotes.

It deliberately avoids the two mistakes that make most home-grown backtests
lie: **look-ahead bias** and **ignoring trading costs**.

## Functions

| Function | Description |
|---|---|
| `to_returns(prices)` | Simple period returns aligned to the price array |
| `run_backtest(prices, target_position, fee_bps, slippage_bps)` | Core engine → equity curve, net/gross returns, costs, turnover |
| `performance_summary(result, periods_per_year)` | CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover |
| `max_drawdown(equity)` | Largest peak-to-trough decline and where it happened |
| `sma_crossover_signal(prices, fast, slow)` | Example long/flat signal to drive the engine |

## How look-ahead is removed

The position you *decide* on bar `t` is only *held* from `t` to `t+1`:

```
held[t] = target[t-1]
pnl[t]  = held[t] * return[t]
```

If you instead let `target[t]` earn `return[t]`, you would be trading on
information you did not have yet — the classic way to manufacture a beautiful,
fake equity curve.

## How costs are charged

Costs are paid on the **traded notional**, i.e. the change in position:

```
traded[t] = |held[t] - held[t-1]|
cost[t]   = traded[t] * (fee_bps + slippage_bps) / 10000
net[t]    = held[t] * return[t] - cost[t]
```

So a full flip (`+1 → -1`) costs twice as much as opening a position
(`0 → +1`), exactly as it should.

## Example

```python
import numpy as np
from backtest_engine import run_backtest, performance_summary, sma_crossover_signal

prices = ...  # your price series
signal = sma_crossover_signal(prices, fast=20, slow=50)

result = run_backtest(prices, signal, fee_bps=1.0, slippage_bps=0.5)
stats = performance_summary(result, periods_per_year=252)

print(stats["sharpe"], stats["max_drawdown"], stats["cagr"])
```

To plug in your own strategy, just pass any `target_position` array — values in
`{-1, 0, 1}` for long/flat/short, or continuous weights for sizing.

## Performance metrics, briefly

- **CAGR** — geometric annual growth rate of the equity curve.
- **Sharpe** — annualised mean return ÷ annualised volatility (excess over a
  zero risk-free rate here).
- **Sortino** — like Sharpe but only penalises *downside* deviation.
- **Max drawdown** — worst peak-to-trough loss; the number that ends careers.
- **Calmar** — CAGR ÷ |max drawdown|; return per unit of pain.
- **Hit rate** — fraction of bars with a positive net return.

## Practical notes

- A backtest is a hypothesis, not a guarantee. Costs, slippage and survivorship
  bias all erode live results versus paper.
- Always compare against **buy & hold** — the `__main__` demo does this. If your
  clever strategy cannot beat holding the asset after costs, it is not clever.
- Vectorised backtests assume you can transact at the marked price. For
  realistic fills on large size, see
  `Order Execution Simulator` and `Finance - Transaction Cost Analysis`.
- Beware overfitting: tuning `fast`/`slow` until the curve looks great is
  curve-fitting. Validate out-of-sample (see
  `Quantitative Methods - Bootstrap`).

---

## Running it

```bash
cd "Strategies - Backtesting Engine"
python "backtest_engine.py"
```

## More in Strategies

- [Order Execution Simulator](Order Execution Simulator.md): This tool lets you simulate buy and sell orders, track a virtual portfolio, and analyze trade performance.
- [Strategies - Market Making](Strategies - Market Making.md): Implementation of the Avellaneda-Stoikov (2008) continuous-time market making model.
- [Strategies - Mean Reversion](Strategies - Mean Reversion.md): Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it.
- [Strategies - Momentum Trading](Strategies - Momentum Trading.md): Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market.
- [Strategies - Optimal Execution](Strategies - Optimal Execution.md): A signal tells you what to own. It does not tell you how to get there.
- [Strategies - Pairs Trading](Strategies - Pairs Trading.md): This module demonstrates a statistical arbitrage strategy known as Pairs Trading.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
