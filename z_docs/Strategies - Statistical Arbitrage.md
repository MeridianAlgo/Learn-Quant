# Strategies - Statistical Arbitrage

<p class="lq-meta">Advanced <span>/</span> Strategies <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Strategies%20-%20Statistical%20Arbitrage">source</a></p>


This module demonstrates a basic Statistical Arbitrage strategy, specifically pairs trading.

## Concepts
- **Cointegration:** A statistical property of a collection of time series variables. If two or more series are themselves non-stationary, but a linear combination of them is stationary, then the series are said to be cointegrated.
- **Pairs Trading:** Identifying two cointegrated assets and trading the spread between them. When the spread widens beyond a historical norm (measured by z-score), you short the outperforming asset and long the underperforming one, betting the spread will revert to the mean.

## Example
Run `python stat_arb.py` to see a demonstration of testing cointegration and generating entry/exit signals based on the z-score of the spread.

---

## Running it

```bash
cd "Strategies - Statistical Arbitrage"
python "stat_arb.py"
```

## More in Strategies

- [Order Execution Simulator](Order Execution Simulator.md): This tool lets you simulate buy and sell orders, track a virtual portfolio, and analyze trade performance.
- [Strategies - Backtesting Engine](Strategies - Backtesting Engine.md): A backtest answers one question: *if I had traded this rule, what would have happened?* This module is a compact, honest backtester for single-asset strategies — it converts a price series and a position series into an...
- [Strategies - Market Making](Strategies - Market Making.md): Implementation of the Avellaneda-Stoikov (2008) continuous-time market making model.
- [Strategies - Mean Reversion](Strategies - Mean Reversion.md): Mean reversion is the statistical tendency for an asset's price to return to its historical average after deviating from it.
- [Strategies - Momentum Trading](Strategies - Momentum Trading.md): Momentum trading is a strategy that capitalizes on the continuance of existing trends in the market.
- [Strategies - Optimal Execution](Strategies - Optimal Execution.md): A signal tells you what to own. It does not tell you how to get there.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
